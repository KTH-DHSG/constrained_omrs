from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import rclpy
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from constrained_omrs import (
    ControllerConfig,
    FormationManager,
    GeometricQuadrotorController,
    QuadrotorConfig,
    QuadrotorRotorActuator,
    SmoothBacksteppingController,
    default_open_formation_scenario,
    motor_speed_for_hover,
    odometry_body_twist_to_world_linear_velocity,
    quaternion_to_rotation_xyzw,
    rotor_thrusts_to_motor_speeds,
)


@dataclass
class VehicleState:
    position: np.ndarray
    velocity_world: np.ndarray
    rotation_world_from_body: np.ndarray
    angular_velocity_body: np.ndarray
    received: bool = False
    last_receive_time: float = -np.inf


class OpenTeamGazeboController(Node):
    """Run the paper controller against Gazebo Harmonic quadrotor dynamics.

    Gazebo owns rigid-body/contact dynamics and the first-order rotor dynamics
    through ``MulticopterMotorModel``. This node only performs the supervisory
    OMRS logic, the BLF/backstepping coordination law, geometric attitude/thrust
    realization, and bounded rotor allocation.
    """

    def __init__(self) -> None:
        super().__init__("open_team_gazebo_controller")

        self.declare_parameter("control_rate_hz", 100.0)
        self.declare_parameter("marker_rate_hz", 20.0)
        self.declare_parameter("odom_timeout", 0.25)
        self.declare_parameter("motor_constant", 7.0 / (900.0**2))
        self.declare_parameter("maximum_rotor_velocity", 900.0)
        self.declare_parameter("startup_hover", True)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.marker_rate_hz = float(self.get_parameter("marker_rate_hz").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.motor_constant = float(self.get_parameter("motor_constant").value)
        self.maximum_rotor_velocity = float(
            self.get_parameter("maximum_rotor_velocity").value
        )
        self.startup_hover = bool(self.get_parameter("startup_hover").value)

        if self.control_rate_hz <= 0.0 or self.marker_rate_hz <= 0.0:
            raise ValueError("control_rate_hz and marker_rate_hz must be positive.")
        if self.motor_constant <= 0.0 or self.maximum_rotor_velocity <= 0.0:
            raise ValueError("motor parameters must be positive.")

        self.scenario = default_open_formation_scenario(3)
        self.num_robots = self.scenario.num_robots
        self.manager = FormationManager(self.scenario)
        self.high_level = SmoothBacksteppingController(
            num_robots=self.num_robots,
            dimension=3,
            collision_distance=self.scenario.collision_distance,
            sensing_distance=self.scenario.sensing_distance,
            config=ControllerConfig(k1=0.5, k2=0.5, k3=2.0, max_virtual_speed=0.6),
        )

        # Gazebo's motor plugin provides its own rotor first-order dynamics, so
        # linear_drag is kept at zero in this realization adapter. The vehicle
        # mass, inertia, geometry, rotor bounds and yaw moment ratio match the
        # SDF model installed with this package.
        self.quadrotor_config = QuadrotorConfig(
            linear_drag=0.0,
            motor_time_constant=0.045,
            yaw_torque_per_thrust=0.018,
        )
        self.geometric_controller = GeometricQuadrotorController(
            self.quadrotor_config
        )
        self.rotor_allocator = QuadrotorRotorActuator(self.quadrotor_config)
        self.hover_speed = motor_speed_for_hover(
            self.quadrotor_config.mass,
            self.quadrotor_config.gravity,
            self.motor_constant,
        )

        self.states = [
            VehicleState(
                position=np.zeros(3),
                velocity_world=np.zeros(3),
                rotation_world_from_body=np.eye(3),
                angular_velocity_body=np.zeros(3),
            )
            for _ in range(self.num_robots)
        ]

        self.motor_publishers = []
        self.odom_subscriptions = []
        for robot in range(self.num_robots):
            self.motor_publishers.append(
                self.create_publisher(
                    Actuators,
                    f"/drone_{robot}/command/motor_speed",
                    10,
                )
            )
            self.odom_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    f"/drone_{robot}/odom",
                    lambda msg, robot=robot: self._odom_callback(robot, msg),
                    20,
                )
            )

        self.marker_publisher = self.create_publisher(
            MarkerArray, "/omrs/markers", 10
        )
        self.status_publisher = self.create_publisher(String, "/omrs/status", 10)

        self.start_sim_time: float | None = None
        self.last_control_time: float | None = None
        self.last_marker_time = -np.inf
        self.last_waiting_log_time = -np.inf
        self.handled_switch_events = 0
        self.returning = np.zeros(self.num_robots, dtype=bool)
        self.parked = ~self.scenario.initial_active.copy()

        period = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(period, self._control_tick)

        self.get_logger().info(
            "Constrained OMRS Gazebo controller initialized for "
            f"{self.num_robots} quadrotors at {self.control_rate_hz:.1f} Hz."
        )
        self.get_logger().info(
            "Waiting for bridged Gazebo odometry. Initial active vehicles receive "
            "hover motor speed while the state interface comes online."
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom_callback(self, robot: int, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        rotation = quaternion_to_rotation_xyzw(
            np.array([q.x, q.y, q.z, q.w], dtype=float)
        )
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        omega = msg.twist.twist.angular

        state = self.states[robot]
        state.position = np.array([p.x, p.y, p.z], dtype=float)
        state.rotation_world_from_body = rotation
        state.velocity_world = odometry_body_twist_to_world_linear_velocity(
            rotation,
            np.array([v.x, v.y, v.z], dtype=float),
        )
        state.angular_velocity_body = np.array(
            [omega.x, omega.y, omega.z], dtype=float
        )
        state.received = True
        state.last_receive_time = self._now_seconds()

    def _publish_motor_speeds(self, robot: int, speeds: np.ndarray) -> None:
        msg = Actuators()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = np.asarray(speeds, dtype=float).reshape(4).tolist()
        self.motor_publishers[robot].publish(msg)

    def _publish_waiting_commands(self) -> None:
        for robot in range(self.num_robots):
            if self.startup_hover and self.scenario.initial_active[robot]:
                self._publish_motor_speeds(
                    robot, np.full(4, self.hover_speed, dtype=float)
                )
            else:
                self._publish_motor_speeds(robot, np.zeros(4))

    def _all_states_ready(self) -> bool:
        return all(state.received for state in self.states)

    def _state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions = np.vstack([state.position for state in self.states])
        velocities = np.vstack([state.velocity_world for state in self.states])
        rotations = np.stack(
            [state.rotation_world_from_body for state in self.states], axis=0
        )
        angular_velocities = np.vstack(
            [state.angular_velocity_body for state in self.states]
        )
        return positions, velocities, rotations, angular_velocities

    def _handle_new_switch_events(self) -> None:
        new_events = self.manager.switch_events[self.handled_switch_events :]
        for event in new_events:
            self.get_logger().info(
                f"OMRS switch at t={event.time:.2f} s: {event.kind}; "
                f"robot={event.robot}; added={list(event.edges_added)}; "
                f"removed={list(event.edges_removed)}"
            )
            if event.kind == "robot_leave" and event.robot is not None:
                self.returning[event.robot] = True
                self.parked[event.robot] = False
        self.handled_switch_events = len(self.manager.switch_events)

    def _update_landing_states(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        charging = self.scenario.charging_area
        if charging is None:
            return
        for robot in np.flatnonzero(self.returning).tolist():
            if charging.landed(positions[robot], velocities[robot], robot):
                self.returning[robot] = False
                self.parked[robot] = True
                self.get_logger().info(
                    f"Drone {robot} reached charging pad; motors switched off."
                )

    def _safe_hover_or_off(self) -> None:
        controlled = self.manager.controlled | self.returning
        for robot in range(self.num_robots):
            if controlled[robot] and not self.parked[robot]:
                self._publish_motor_speeds(
                    robot, np.full(4, self.hover_speed, dtype=float)
                )
            else:
                self._publish_motor_speeds(robot, np.zeros(4))

    def _control_tick(self) -> None:
        now = self._now_seconds()
        if not self._all_states_ready():
            self._publish_waiting_commands()
            if now - self.last_waiting_log_time > 2.0:
                missing = [i for i, state in enumerate(self.states) if not state.received]
                self.get_logger().info(f"Waiting for odometry from drones {missing}.")
                self.last_waiting_log_time = now
            return

        if self.start_sim_time is None:
            self.start_sim_time = now
            self.last_control_time = now
            self.get_logger().info("All vehicle states received; starting OMRS mission clock.")

        assert self.last_control_time is not None
        scenario_time = now - self.start_sim_time
        dt = now - self.last_control_time
        if dt <= 0.0:
            return

        positions, velocities, rotations, angular_velocities = self._state_arrays()

        # If any state stream disappears after takeoff, do not continue evolving
        # the supervisory relaxation state using stale data. Hover is the least
        # destructive simulation fallback while retaining a clear diagnostic.
        stale = [
            i
            for i, state in enumerate(self.states)
            if now - state.last_receive_time > self.odom_timeout
        ]
        if stale:
            self.get_logger().error(
                f"Stale Gazebo odometry for drones {stale}; commanding hover/off fallback."
            )
            self._safe_hover_or_off()
            self.last_control_time = now
            return

        try:
            # Integrate the previous prospective-edge rates over the elapsed
            # physical time, using the current Gazebo positions as the endpoint
            # for the small discrete feasibility projection.
            if self.manager.prospective:
                self.manager.integrate_relaxations(dt, positions)

            self.manager.process(scenario_time, positions, velocities)
            self.manager.assert_established_graph_connected()
            self._handle_new_switch_events()
            self._update_landing_states(positions, velocities)

            formation_controlled = self.manager.controlled.copy()
            newly_controlled = formation_controlled & self.parked
            self.parked[newly_controlled] = False
            flight_controlled = formation_controlled | self.returning

            interactions = self.manager.controller_interactions()
            high_level = self.high_level.control(
                positions,
                velocities,
                interactions,
                formation_controlled,
            )
            acceleration_commands = np.asarray(high_level["u"], dtype=float)

            charging = self.scenario.charging_area
            if charging is not None:
                for robot in np.flatnonzero(self.returning):
                    acceleration_commands[robot] = charging.acceleration_command(
                        positions[robot], velocities[robot], int(robot)
                    )

            for robot in range(self.num_robots):
                if not flight_controlled[robot] or self.parked[robot]:
                    self._publish_motor_speeds(robot, np.zeros(4))
                    continue

                low_level = self.geometric_controller.evaluate(
                    rotations[robot],
                    angular_velocities[robot],
                    velocities[robot],
                    acceleration_commands[robot],
                )
                allocation = self.rotor_allocator.allocate(
                    low_level.thrust,
                    low_level.torque,
                )
                motor_speeds = rotor_thrusts_to_motor_speeds(
                    allocation.commands,
                    self.motor_constant,
                    self.maximum_rotor_velocity,
                )
                self._publish_motor_speeds(robot, motor_speeds)

            if now - self.last_marker_time >= 1.0 / self.marker_rate_hz:
                self._publish_markers(positions)
                self._publish_status(scenario_time)
                self.last_marker_time = now

        except (FloatingPointError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Controller exception: {exc}")
            self._safe_hover_or_off()

        self.last_control_time = now

    @staticmethod
    def _point(vector: np.ndarray) -> Point:
        p = Point()
        p.x, p.y, p.z = [float(value) for value in vector]
        return p

    def _desired_positions_for_visualization(self, positions: np.ndarray) -> np.ndarray:
        desired = self.manager.current_desired_positions.copy()
        active_ids = np.flatnonzero(self.manager.active)
        if active_ids.size:
            translation = (
                np.mean(positions[active_ids], axis=0)
                - np.mean(desired[active_ids], axis=0)
            )
            desired = desired + translation
        return desired

    def _publish_markers(self, positions: np.ndarray) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        stamp = self.get_clock().now().to_msg()

        desired = self._desired_positions_for_visualization(positions)
        desired_marker = Marker()
        desired_marker.header.frame_id = "world"
        desired_marker.header.stamp = stamp
        desired_marker.ns = "desired_positions"
        desired_marker.id = 0
        desired_marker.type = Marker.SPHERE_LIST
        desired_marker.action = Marker.ADD
        desired_marker.scale.x = 0.10
        desired_marker.scale.y = 0.10
        desired_marker.scale.z = 0.10
        desired_marker.color.r = 0.95
        desired_marker.color.g = 0.75
        desired_marker.color.b = 0.15
        desired_marker.color.a = 0.75
        desired_marker.points = [
            self._point(desired[i]) for i in np.flatnonzero(self.manager.controlled)
        ]
        markers.markers.append(desired_marker)

        established = Marker()
        established.header.frame_id = "world"
        established.header.stamp = stamp
        established.ns = "established_edges"
        established.id = 0
        established.type = Marker.LINE_LIST
        established.action = Marker.ADD
        established.scale.x = 0.025
        established.color.r = 0.10
        established.color.g = 0.75
        established.color.b = 0.95
        established.color.a = 0.90
        for i, j in sorted(self.manager.established_edges):
            established.points.extend([self._point(positions[i]), self._point(positions[j])])
        markers.markers.append(established)

        prospective = Marker()
        prospective.header.frame_id = "world"
        prospective.header.stamp = stamp
        prospective.ns = "prospective_edges"
        prospective.id = 0
        prospective.type = Marker.LINE_LIST
        prospective.action = Marker.ADD
        prospective.scale.x = 0.025
        prospective.color.r = 0.95
        prospective.color.g = 0.45
        prospective.color.b = 0.10
        prospective.color.a = 0.90
        dash_count = 10
        for edge, state in sorted(self.manager.prospective.items()):
            i, j = edge
            p0 = positions[i]
            p1 = positions[j]
            for segment in range(0, dash_count, 2):
                alpha0 = segment / dash_count
                alpha1 = (segment + 1) / dash_count
                prospective.points.extend(
                    [
                        self._point((1.0 - alpha0) * p0 + alpha0 * p1),
                        self._point((1.0 - alpha1) * p0 + alpha1 * p1),
                    ]
                )

            text = Marker()
            text.header.frame_id = "world"
            text.header.stamp = stamp
            text.ns = "prospective_rho"
            text.id = 100 + 10 * edge[0] + edge[1]
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            midpoint = 0.5 * (p0 + p1)
            text.pose.position = self._point(midpoint)
            text.scale.z = 0.12
            text.color.r = 0.95
            text.color.g = 0.45
            text.color.b = 0.10
            text.color.a = 1.0
            text.text = (
                f"({i},{j})  dmax+rho="
                f"{self.scenario.sensing_distance + state.rho:.2f} m"
            )
            markers.markers.append(text)
        markers.markers.append(prospective)

        if self.scenario.charging_area is not None:
            pads = self.scenario.charging_area.pad_positions
            platform = Marker()
            platform.header.frame_id = "world"
            platform.header.stamp = stamp
            platform.ns = "charging_area"
            platform.id = 0
            platform.type = Marker.CUBE
            platform.action = Marker.ADD
            center = 0.5 * (np.min(pads, axis=0) + np.max(pads, axis=0))
            platform.pose.position = self._point(
                np.array([center[0], center[1], 0.015], dtype=float)
            )
            platform.pose.orientation.w = 1.0
            platform.scale.x = float(np.ptp(pads[:, 0]) + 0.75)
            platform.scale.y = float(np.ptp(pads[:, 1]) + 0.75)
            platform.scale.z = 0.03
            platform.color.r = 0.20
            platform.color.g = 0.20
            platform.color.b = 0.22
            platform.color.a = 0.35
            markers.markers.append(platform)

        self.marker_publisher.publish(markers)

    def _publish_status(self, scenario_time: float) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "time": round(float(scenario_time), 3),
                "active": np.flatnonzero(self.manager.active).tolist(),
                "staging": np.flatnonzero(self.manager.staging).tolist(),
                "returning": np.flatnonzero(self.returning).tolist(),
                "parked": np.flatnonzero(self.parked).tolist(),
                "established_edges": [list(edge) for edge in sorted(self.manager.established_edges)],
                "prospective_edges": [
                    {
                        "edge": list(edge),
                        "rho": round(float(state.rho), 4),
                        "relaxed_dmax": round(
                            float(self.scenario.sensing_distance + state.rho), 4
                        ),
                    }
                    for edge, state in sorted(self.manager.prospective.items())
                ],
            }
        )
        self.status_publisher.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OpenTeamGazeboController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for robot in range(node.num_robots):
            node._publish_motor_speeds(robot, np.zeros(4))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
