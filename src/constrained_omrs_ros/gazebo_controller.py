from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from threading import Thread
from time import perf_counter

import numpy as np
import rclpy
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from constrained_omrs_ros.gazebo_markers import GazeboEdgeMarkerClient
from constrained_omrs_ros.gazebo_results import (
    GazeboMissionRecorder,
    create_run_directory,
    save_raw_history,
    save_diagnostic_csvs,
)

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
    last_state_time: float = -np.inf


class OpenTeamGazeboController(Node):
    """Run the OMRS controller against Gazebo quadrotor dynamics.

    A service-flight state machine handles motor spool-up, takeoff, initial
    deployment, admission from the charging station, and return-to-charge.
    The OMRS mission clock starts only after the initially active team has
    reached a valid airborne initial condition.  These service phases are
    intentionally outside the paper's open-system dynamics.
    """

    def __init__(self) -> None:
        super().__init__("open_team_gazebo_controller")

        self.declare_parameter("control_rate_hz", 100.0)
        self.declare_parameter("marker_rate_hz", 10.0)
        self.declare_parameter("odom_timeout", 0.25)
        self.declare_parameter("motor_constant", 7.0 / (900.0**2))
        self.declare_parameter("maximum_rotor_velocity", 900.0)
        self.declare_parameter("motor_spool_time", 1.0)
        self.declare_parameter("initial_takeoff_stagger", 0.75)
        self.declare_parameter("takeoff_height", 0.85)
        self.declare_parameter("service_maximum_speed", 0.45)
        self.declare_parameter("service_maximum_acceleration", 0.90)
        self.declare_parameter("service_position_gain", 0.45)
        self.declare_parameter("service_velocity_gain", 2.20)
        self.declare_parameter("service_position_tolerance", 0.15)
        self.declare_parameter("service_speed_tolerance", 0.12)
        self.declare_parameter("takeoff_settle_time", 0.80)
        self.declare_parameter("service_settle_time", 1.20)
        # A joining vehicle first approaches the existing formation under the
        # service controller. Prospective edges are created only once every
        # edge that will be selected lies moderately outside (or inside) the
        # nominal sensing radius. This prevents a far-away joining robot from
        # pulling the established team through a very large prospective BLF.
        self.declare_parameter("join_capture_margin", 0.50)
        self.declare_parameter("join_capture_lower_margin", 0.20)
        # Capture is based on the prospective relative-vector error as well as
        # scalar distance.  The previous distance-only gate could trigger while
        # the candidate was on the wrong side of its neighbors, with edge
        # errors around 1.8 m despite admissible pairwise distances.
        self.declare_parameter("join_capture_error_tolerance", 1.10)
        self.declare_parameter("join_capture_speed_tolerance", 0.45)
        self.declare_parameter("diagnostic_log_period", 1.0)
        self.declare_parameter("attitude_gain", 1.8)
        self.declare_parameter("angular_velocity_gain", 0.50)
        self.declare_parameter("maximum_torque", 0.70)
        self.declare_parameter("maximum_tilt_deg", 25.0)

        # Gazebo-specific shaping of the paper-level formation controller.
        # Keep only a magnitude limit by default. A slow acceleration slew
        # limiter was found to introduce a large phase lag: during ordinary
        # fixed-team formation convergence it could keep applying an old
        # acceleration after the backstepping controller had already reversed
        # direction. Smooth topology handovers are handled explicitly instead.
        self.declare_parameter("formation_k1", 0.30)
        self.declare_parameter("formation_k2", 0.30)
        self.declare_parameter("formation_k3", 1.25)
        self.declare_parameter("formation_max_virtual_speed", 0.35)
        self.declare_parameter("formation_collision_weight", 0.02)
        self.declare_parameter("formation_max_edge_damping", 0.22)
        self.declare_parameter("formation_max_local_damping", 0.38)
        self.declare_parameter("formation_maximum_acceleration", 0.85)
        self.declare_parameter("formation_acceleration_rate_limit", 0.0)

        # Prospective-edge relaxation / handover parameters for the physical
        # Gazebo plant. rho contracts at a constant nominal rate and is
        # initialized with a generous protected margin. Activation occurs at
        # rho == 0. The post-switch physical controller is introduced through
        # a slow whole-controller homotopy rather than an internal graph-matrix
        # interpolation.
        self.declare_parameter("rho_initial_margin", 0.22)
        self.declare_parameter("rho_contraction_rate", 0.10)
        self.declare_parameter("activation_distance_margin", 0.20)
        self.declare_parameter("activation_outward_rate_tolerance", 0.08)
        self.declare_parameter("join_handover_duration", 6.0)
        # Keep a joining interaction prospective for long enough that the
        # candidate-only BLF controller is genuinely exercised before
        # admission, even when rho happens to be zero at assignment.
        self.declare_parameter("join_prospective_min_duration", 2.5)
        self.declare_parameter("join_activation_error_tolerance", 0.45)
        self.declare_parameter("switch_diagnostic_period", 0.10)

        # Mission completion / automatic result generation.
        self.declare_parameter("completion_position_tolerance", 0.12)
        self.declare_parameter("completion_velocity_tolerance", 0.12)
        self.declare_parameter("completion_settle_time", 2.0)
        self.declare_parameter("stop_on_mission_complete", True)
        self.declare_parameter("return_to_charge_on_completion", True)
        self.declare_parameter("save_results", True)
        self.declare_parameter("save_animation", True)
        self.declare_parameter("result_sample_rate_hz", 20.0)
        self.declare_parameter("animation_fps", 15)
        self.declare_parameter("animation_stride", 2)
        self.declare_parameter("output_directory", "package_outputs")
        self.declare_parameter("gazebo_edge_markers", True)
        self.declare_parameter("fixed_team_only", False)
        self.declare_parameter("save_animation_on_interrupt", True)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.marker_rate_hz = float(self.get_parameter("marker_rate_hz").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.motor_constant = float(self.get_parameter("motor_constant").value)
        self.maximum_rotor_velocity = float(self.get_parameter("maximum_rotor_velocity").value)
        self.motor_spool_time = float(self.get_parameter("motor_spool_time").value)
        self.initial_takeoff_stagger = float(self.get_parameter("initial_takeoff_stagger").value)
        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.service_maximum_speed = float(self.get_parameter("service_maximum_speed").value)
        self.service_maximum_acceleration = float(self.get_parameter("service_maximum_acceleration").value)
        self.service_position_gain = float(self.get_parameter("service_position_gain").value)
        self.service_velocity_gain = float(self.get_parameter("service_velocity_gain").value)
        self.service_position_tolerance = float(self.get_parameter("service_position_tolerance").value)
        self.service_speed_tolerance = float(self.get_parameter("service_speed_tolerance").value)
        self.takeoff_settle_time = float(self.get_parameter("takeoff_settle_time").value)
        self.service_settle_time = float(self.get_parameter("service_settle_time").value)
        self.join_capture_margin = float(self.get_parameter("join_capture_margin").value)
        self.join_capture_lower_margin = float(
            self.get_parameter("join_capture_lower_margin").value
        )
        self.join_capture_error_tolerance = float(
            self.get_parameter("join_capture_error_tolerance").value
        )
        self.join_capture_speed_tolerance = float(
            self.get_parameter("join_capture_speed_tolerance").value
        )
        self.diagnostic_log_period = float(self.get_parameter("diagnostic_log_period").value)
        self.attitude_gain = float(self.get_parameter("attitude_gain").value)
        self.angular_velocity_gain = float(self.get_parameter("angular_velocity_gain").value)
        self.maximum_torque = float(self.get_parameter("maximum_torque").value)
        self.maximum_tilt_deg = float(self.get_parameter("maximum_tilt_deg").value)
        self.formation_k1 = float(self.get_parameter("formation_k1").value)
        self.formation_k2 = float(self.get_parameter("formation_k2").value)
        self.formation_k3 = float(self.get_parameter("formation_k3").value)
        self.formation_max_virtual_speed = float(self.get_parameter("formation_max_virtual_speed").value)
        self.formation_collision_weight = float(self.get_parameter("formation_collision_weight").value)
        self.formation_max_edge_damping = float(self.get_parameter("formation_max_edge_damping").value)
        self.formation_max_local_damping = float(self.get_parameter("formation_max_local_damping").value)
        self.formation_maximum_acceleration = float(self.get_parameter("formation_maximum_acceleration").value)
        self.formation_acceleration_rate_limit = float(self.get_parameter("formation_acceleration_rate_limit").value)
        self.rho_initial_margin = float(self.get_parameter("rho_initial_margin").value)
        self.rho_contraction_rate = float(self.get_parameter("rho_contraction_rate").value)
        self.activation_distance_margin = float(self.get_parameter("activation_distance_margin").value)
        self.activation_outward_rate_tolerance = float(
            self.get_parameter("activation_outward_rate_tolerance").value
        )
        self.join_handover_duration = float(
            self.get_parameter("join_handover_duration").value
        )
        self.join_prospective_min_duration = float(
            self.get_parameter("join_prospective_min_duration").value
        )
        self.join_activation_error_tolerance = float(
            self.get_parameter("join_activation_error_tolerance").value
        )
        self.switch_diagnostic_period = float(
            self.get_parameter("switch_diagnostic_period").value
        )
        self.completion_position_tolerance = float(self.get_parameter("completion_position_tolerance").value)
        self.completion_velocity_tolerance = float(self.get_parameter("completion_velocity_tolerance").value)
        self.completion_settle_time = float(self.get_parameter("completion_settle_time").value)
        self.stop_on_mission_complete = bool(self.get_parameter("stop_on_mission_complete").value)
        self.return_to_charge_on_completion = bool(
            self.get_parameter("return_to_charge_on_completion").value
        )
        self.save_results = bool(self.get_parameter("save_results").value)
        self.save_animation = bool(self.get_parameter("save_animation").value)
        self.result_sample_rate_hz = float(self.get_parameter("result_sample_rate_hz").value)
        self.animation_fps = int(self.get_parameter("animation_fps").value)
        self.animation_stride = int(self.get_parameter("animation_stride").value)
        self.fixed_team_only = bool(self.get_parameter("fixed_team_only").value)
        self.save_animation_on_interrupt = bool(self.get_parameter("save_animation_on_interrupt").value)
        self.output_directory = str(self.get_parameter("output_directory").value)
        self.gazebo_edge_markers_enabled = bool(self.get_parameter("gazebo_edge_markers").value)

        for name in (
            "control_rate_hz", "marker_rate_hz", "odom_timeout", "motor_constant",
            "maximum_rotor_velocity", "motor_spool_time", "takeoff_height",
            "service_maximum_speed", "service_maximum_acceleration",
            "service_position_gain", "service_velocity_gain",
            "service_position_tolerance", "service_speed_tolerance",
            "takeoff_settle_time", "service_settle_time",
            "join_capture_margin", "join_capture_error_tolerance",
            "join_capture_speed_tolerance",
            "diagnostic_log_period",
            "attitude_gain", "angular_velocity_gain", "maximum_torque",
            "maximum_tilt_deg",
            "formation_k1", "formation_k2", "formation_k3",
            "formation_max_virtual_speed", "formation_collision_weight",
            "formation_max_edge_damping", "formation_max_local_damping",
            "formation_maximum_acceleration",
            "rho_initial_margin",
            "rho_contraction_rate", "activation_distance_margin",
            "join_activation_error_tolerance",
            "switch_diagnostic_period",
            "completion_position_tolerance",
            "completion_velocity_tolerance", "completion_settle_time",
            "result_sample_rate_hz",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.initial_takeoff_stagger < 0.0:
            raise ValueError("initial_takeoff_stagger must be nonnegative.")
        if self.join_capture_margin < 0.0 or self.join_capture_lower_margin < 0.0:
            raise ValueError("join capture margins must be nonnegative.")
        if self.formation_acceleration_rate_limit < 0.0:
            raise ValueError("formation_acceleration_rate_limit must be nonnegative.")
        if self.activation_outward_rate_tolerance < 0.0:
            raise ValueError("activation_outward_rate_tolerance must be nonnegative.")
        if self.join_handover_duration < 0.0:
            raise ValueError("join_handover_duration must be nonnegative.")
        if self.join_prospective_min_duration < 0.0:
            raise ValueError("join_prospective_min_duration must be nonnegative.")

        self.scenario = default_open_formation_scenario(3)
        if self.fixed_team_only:
            self.scenario = replace(self.scenario, join_requests=tuple(), leave_requests=tuple())
        # Keep the pure-Python scenario / theory defaults unchanged, but use a
        # more conservative switch guard for the physical Gazebo realization.
        self.scenario = replace(
            self.scenario,
            reconfiguration=replace(
                self.scenario.reconfiguration,
                rho_initial_margin=self.rho_initial_margin,
                rho_contraction_rate=self.rho_contraction_rate,
                edge_activation_margin=self.activation_distance_margin,
                edge_activation_outward_rate_tolerance=self.activation_outward_rate_tolerance,
                join_handover_duration=self.join_handover_duration,
                join_prospective_min_duration=self.join_prospective_min_duration,
                join_activation_error_tolerance=self.join_activation_error_tolerance,
            ),
        )
        if self.scenario.charging_area is None:
            raise RuntimeError("The Gazebo experiment requires a charging area.")
        self.num_robots = self.scenario.num_robots
        self.manager = FormationManager(self.scenario)
        self.high_level = SmoothBacksteppingController(
            num_robots=self.num_robots,
            dimension=3,
            collision_distance=self.scenario.collision_distance,
            sensing_distance=self.scenario.sensing_distance,
            config=ControllerConfig(
                k1=self.formation_k1,
                k2=self.formation_k2,
                k3=self.formation_k3,
                max_virtual_speed=self.formation_max_virtual_speed,
                collision_weight=self.formation_collision_weight,
                max_edge_damping=self.formation_max_edge_damping,
                max_local_damping=self.formation_max_local_damping,
            ),
        )

        # Gazebo applies yaw reaction torque opposite to rotor turning direction
        # (dragTorque = -turningDirection * thrust * momentConstant).  Therefore
        # the yaw-wrench signs used by the allocator are [-,+,-,+] for the SDF
        # sequence [ccw,cw,ccw,cw].
        self.quadrotor_config = QuadrotorConfig(
            linear_drag=0.0,
            attitude_gain=self.attitude_gain,
            angular_velocity_gain=self.angular_velocity_gain,
            maximum_torque=self.maximum_torque,
            maximum_tilt_deg=self.maximum_tilt_deg,
            motor_time_constant=0.045,
            yaw_torque_per_thrust=0.018,
            rotor_spin_directions=(-1, 1, -1, 1),
        )
        self.geometric_controller = GeometricQuadrotorController(self.quadrotor_config)
        self.rotor_allocator = QuadrotorRotorActuator(self.quadrotor_config)
        self.hover_speed = motor_speed_for_hover(
            self.quadrotor_config.mass,
            self.quadrotor_config.gravity,
            self.motor_constant,
        )

        self.states = [
            VehicleState(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
            for _ in range(self.num_robots)
        ]
        self.motor_publishers = []
        self.odom_subscriptions = []
        for robot in range(self.num_robots):
            self.motor_publishers.append(
                self.create_publisher(Actuators, f"/drone_{robot}/command/motor_speed", 10)
            )
            self.odom_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    f"/drone_{robot}/odom",
                    lambda msg, robot=robot: self._odom_callback(robot, msg),
                    qos_profile_sensor_data,
                )
            )

        self.marker_publisher = self.create_publisher(MarkerArray, "/omrs/markers", 10)
        self.status_publisher = self.create_publisher(String, "/omrs/status", 10)

        self.last_control_time: float | None = None
        self.preflight_start_time: float | None = None
        self.mission_start_time: float | None = None
        self.last_marker_time = -np.inf
        self.last_waiting_log_time = -np.inf
        self.handled_switch_events = 0

        self.returning = np.zeros(self.num_robots, dtype=bool)
        self.parked = np.ones(self.num_robots, dtype=bool)
        self.join_ready = np.zeros(self.num_robots, dtype=bool)
        self.service_phase = ["parked" for _ in range(self.num_robots)]
        self.phase_start_time = np.full(self.num_robots, np.nan, dtype=float)
        self.target_settle_start = np.full(self.num_robots, np.nan, dtype=float)
        self.last_attitude_error_deg = np.zeros(self.num_robots, dtype=float)
        self.last_acceleration_command_norm = np.zeros(self.num_robots, dtype=float)
        self.last_motor_speed_min = np.zeros(self.num_robots, dtype=float)
        self.last_motor_speed_max = np.zeros(self.num_robots, dtype=float)
        self.last_allocation_saturated = np.zeros(self.num_robots, dtype=bool)
        self.last_diagnostic_log_time = -np.inf
        self.last_fixed_team_log_time = -np.inf
        self.last_flight_acceleration_command = np.zeros((self.num_robots, 3), dtype=float)
        self.last_commanded_thrust = np.zeros(self.num_robots, dtype=float)
        self.last_commanded_torque = np.zeros((self.num_robots, 3), dtype=float)

        # Detailed topology-transition diagnostics. Rows are both printed at a
        # high temporal resolution around robot admission and saved with the
        # Gazebo result bundle as join_transition_log.csv.
        self.join_transition_log_rows: list[dict[str, object]] = []
        self.active_join_diagnostic_until = -np.inf
        self.last_switch_diagnostic_time = -np.inf
        self._pending_join_switch_snapshot: dict[str, object] | None = None

        self.mission_complete = False
        self.post_mission_recovery_started = False
        self.completion_candidate_start = np.nan
        self.completion_time: float | None = None
        self._results_thread: Thread | None = None
        self._results_done = False
        self._results_logged = False
        self._results_error: str | None = None
        self._result_paths: list[str] = []
        self._result_directory: str | None = None
        self._shutdown_requested = False
        self.exception_log_rows: list[dict[str, object]] = []
        self.last_exception_snapshot_time = -np.inf

        # Runtime timing diagnostics are deliberately based on both simulation
        # time and wall time.  ``sim_dt`` reveals missed / coalesced controller
        # ticks, while ``local_real_time_factor`` quantifies how much slower
        # Gazebo is running in wall time (e.g. while screen recording).
        self.runtime_performance_rows: list[dict[str, float | str]] = []
        self._last_control_wall_start: float | None = None
        self._last_control_sim_start: float | None = None

        self.recorder = GazeboMissionRecorder(
            self.scenario, self.quadrotor_config, self.result_sample_rate_hz
        )

        self.gz_edge_markers = GazeboEdgeMarkerClient() if self.gazebo_edge_markers_enabled else None
        self._gz_marker_success_logged = False
        self._gz_marker_failure_logged = False
        # Native Gazebo marker transport is intentionally kept out of the
        # control callback.  MarkerManager is a GUI-facing synchronous service
        # and can become slow while recording the screen.  The background
        # worker drops intermediate visualization frames rather than ever
        # delaying flight control.
        self._gz_marker_thread: Thread | None = None
        self._gz_marker_result: tuple[bool, str | None] | None = None
        self.last_collision_pairs: list[tuple[int, int]] = []
        if self.gz_edge_markers is not None:
            if self.gz_edge_markers.available:
                self.get_logger().info(
                    "Native Gazebo edge visualization enabled (thick dashed green established, dashed orange prospective)."
                )
            else:
                self.get_logger().warning(
                    "Gazebo edge markers disabled because Gazebo Python bindings are unavailable: "
                    f"{self.gz_edge_markers.error}. Install python3-gz-transport13 and "
                    "python3-gz-msgs10 to enable them. RViz markers remain available."
                )

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._control_tick)
        self.get_logger().info(
            f"Constrained OMRS Gazebo controller initialized for {self.num_robots} "
            f"quadrotors at {self.control_rate_hz:.1f} Hz."
        )
        self.get_logger().info(
            "Waiting for Gazebo odometry. All vehicles remain on their charging "
            "pads with motors off until state feedback is available."
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom_callback(self, robot: int, msg: Odometry) -> None:
        now = self._now_seconds()
        stamp = msg.header.stamp
        state_time = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        # Gazebo odometry normally carries the simulation timestamp.  Keep a
        # receive-time fallback for startup / unusual bridges that emit a zero
        # stamp, but never let an older queued message overwrite newer state.
        if state_time <= 0.0:
            state_time = now

        state = self.states[robot]
        if state.received and state_time + 1e-9 < state.last_state_time:
            return

        q = msg.pose.pose.orientation
        rotation = quaternion_to_rotation_xyzw(np.array([q.x, q.y, q.z, q.w], dtype=float))
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        omega = msg.twist.twist.angular
        state.position = np.array([p.x, p.y, p.z], dtype=float)
        state.rotation_world_from_body = rotation
        state.velocity_world = odometry_body_twist_to_world_linear_velocity(
            rotation, np.array([v.x, v.y, v.z], dtype=float)
        )
        state.angular_velocity_body = np.array([omega.x, omega.y, omega.z], dtype=float)
        state.received = True
        state.last_receive_time = now
        state.last_state_time = state_time

    def _publish_motor_speeds(self, robot: int, speeds: np.ndarray) -> None:
        msg = Actuators()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = np.asarray(speeds, dtype=float).reshape(4).tolist()
        self.motor_publishers[robot].publish(msg)

    def _publish_all_off(self) -> None:
        for robot in range(self.num_robots):
            self._publish_motor_speeds(robot, np.zeros(4))

    def _all_states_ready(self) -> bool:
        return all(state.received for state in self.states)

    def _state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.vstack([state.position for state in self.states]),
            np.vstack([state.velocity_world for state in self.states]),
            np.stack([state.rotation_world_from_body for state in self.states], axis=0),
            np.vstack([state.angular_velocity_body for state in self.states]),
        )

    def _set_phase(self, robot: int, phase: str, now: float) -> None:
        if self.service_phase[robot] == phase:
            return
        self.service_phase[robot] = phase
        self.phase_start_time[robot] = now
        self.target_settle_start[robot] = np.nan
        self.get_logger().info(f"Drone {robot}: service phase -> {phase}.")

    def _publish_acceleration_control(
        self,
        robot: int,
        acceleration_command: np.ndarray,
        rotations: np.ndarray,
        angular_velocities: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        low_level = self.geometric_controller.evaluate(
            rotations[robot], angular_velocities[robot], velocities[robot], acceleration_command
        )
        allocation = self.rotor_allocator.allocate(low_level.thrust, low_level.torque)
        speeds = rotor_thrusts_to_motor_speeds(
            allocation.commands, self.motor_constant, self.maximum_rotor_velocity
        )
        self.last_attitude_error_deg[robot] = float(np.rad2deg(low_level.attitude_error))
        self.last_acceleration_command_norm[robot] = float(
            np.linalg.norm(np.asarray(acceleration_command, dtype=float))
        )
        self.last_motor_speed_min[robot] = float(np.min(speeds))
        self.last_motor_speed_max[robot] = float(np.max(speeds))
        self.last_allocation_saturated[robot] = allocation.saturated
        self.last_flight_acceleration_command[robot] = np.asarray(acceleration_command, dtype=float)
        self.last_commanded_thrust[robot] = float(low_level.thrust)
        self.last_commanded_torque[robot] = np.asarray(low_level.torque, dtype=float)
        self._publish_motor_speeds(robot, speeds)

    def _service_collision_correction(
        self, robot: int, positions: np.ndarray
    ) -> np.ndarray:
        """Collision-barrier correction for service-flight vehicles.

        Service phases are outside the paper-level OMRS controller, but they
        are still physical flights. Every airborne robot therefore participates
        in the same compact lower-distance barrier whenever another airborne
        robot lies inside d_max.
        """
        participants = ~self.parked
        pairs = self.high_level.nearby_collision_pairs(positions, participants)
        if not pairs:
            return np.zeros(3, dtype=float)
        _, _, q_collision, _ = self.high_level.collision_quantities(positions, pairs)
        return -np.asarray(q_collision[robot], dtype=float)

    def _service_command_to_target(
        self,
        robot: int,
        target: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        rotations: np.ndarray,
        angular_velocities: np.ndarray,
    ) -> None:
        charging = self.scenario.charging_area
        assert charging is not None
        command = charging.point_acceleration_command(
            positions[robot], velocities[robot], target,
            maximum_speed=self.service_maximum_speed,
            maximum_acceleration=self.service_maximum_acceleration,
            position_gain=self.service_position_gain,
            velocity_gain=self.service_velocity_gain,
        )
        command = command + self._service_collision_correction(robot, positions)
        command = self._limit_vector_norm(command, self.service_maximum_acceleration)
        self._publish_acceleration_control(
            robot, command, rotations, angular_velocities, velocities
        )

    def _target_reached(
        self, robot: int, target: np.ndarray, positions: np.ndarray, velocities: np.ndarray
    ) -> bool:
        charging = self.scenario.charging_area
        assert charging is not None
        return charging.target_reached(
            positions[robot], velocities[robot], target,
            position_tolerance=self.service_position_tolerance,
            speed_tolerance=self.service_speed_tolerance,
        )

    def _settled_at_target(
        self,
        robot: int,
        target: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        now: float,
        required_time: float,
    ) -> bool:
        if not self._target_reached(robot, target, positions, velocities):
            self.target_settle_start[robot] = np.nan
            return False
        if not np.isfinite(self.target_settle_start[robot]):
            self.target_settle_start[robot] = now
            return required_time <= 0.0
        return now - self.target_settle_start[robot] >= required_time

    def _log_preflight_diagnostics(
        self,
        now: float,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        if now - self.last_diagnostic_log_time < self.diagnostic_log_period:
            return
        charging = self.scenario.charging_area
        assert charging is not None
        parts: list[str] = []
        for robot in np.flatnonzero(self.scenario.initial_active).tolist():
            phase = self.service_phase[robot]
            if phase == "takeoff":
                target = charging.takeoff_waypoint(robot, self.takeoff_height)
            elif phase in ("deploy", "ready"):
                target = self.scenario.initial_positions[robot]
            else:
                continue
            error = float(np.linalg.norm(target - positions[robot]))
            speed = float(np.linalg.norm(velocities[robot]))
            saturation = " sat" if self.last_allocation_saturated[robot] else ""
            parts.append(
                f"r{robot} {phase}: |e|={error:.2f} m, |v|={speed:.2f} m/s, "
                f"att={self.last_attitude_error_deg[robot]:.1f} deg, "
                f"|a_c|={self.last_acceleration_command_norm[robot]:.2f}, "
                f"omega=[{self.last_motor_speed_min[robot]:.0f},"
                f"{self.last_motor_speed_max[robot]:.0f}]{saturation}"
            )
        if parts:
            self.get_logger().info("Preflight convergence -- " + "; ".join(parts))
        self.last_diagnostic_log_time = now

    def _log_fixed_team_formation_diagnostics(
        self,
        scenario_time: float,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        """Report the fixed-team formation transient before the first join.

        The richer initial condition is intentional: this diagnostic makes it
        easy to verify that the nominal formation controller converges stably
        before any open-team event is allowed to complicate the experiment.
        """
        first_join_time = (
            float(self.scenario.join_requests[0].time)
            if self.scenario.join_requests
            else float("inf")
        )
        if scenario_time >= first_join_time:
            return
        if scenario_time - self.last_fixed_team_log_time < self.diagnostic_log_period - 1e-9:
            return
        errors = []
        edge_distances = []
        for i, j in sorted(self.manager.established_edges):
            desired = self.manager.desired_relative_for_edge((i, j))
            errors.append(float(np.linalg.norm(positions[i] - positions[j] - desired)))
            edge_distances.append(float(np.linalg.norm(positions[i] - positions[j])))
        error_norm = float(np.linalg.norm(errors)) if errors else 0.0
        max_edge = max(edge_distances) if edge_distances else float("nan")
        active_ids = np.flatnonzero(self.manager.active)
        max_speed = (
            float(np.max(np.linalg.norm(velocities[active_ids], axis=1)))
            if active_ids.size
            else 0.0
        )
        self.get_logger().info(
            "FIXED-TEAM FORMATION t=%.2f s: ||e_E||=%.3f m, "
            "max edge distance=%.3f m, max speed=%.3f m/s"
            % (scenario_time, error_norm, max_edge, max_speed)
        )
        self.last_fixed_team_log_time = float(scenario_time)

    def _spool_command(self, robot: int, now: float) -> None:
        start = self.phase_start_time[robot]
        alpha = np.clip((now - start) / self.motor_spool_time, 0.0, 1.0)
        # A smooth ramp avoids an impulsive contact release while the vehicle is
        # still supported by the pad.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        self._publish_motor_speeds(robot, np.full(4, alpha * self.hover_speed))

    def _run_initial_deployment(
        self,
        now: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        rotations: np.ndarray,
        angular_velocities: np.ndarray,
    ) -> bool:
        charging = self.scenario.charging_area
        assert charging is not None
        assert self.preflight_start_time is not None

        for order, robot in enumerate(np.flatnonzero(self.scenario.initial_active).tolist()):
            release_time = self.preflight_start_time + order * self.initial_takeoff_stagger
            phase = self.service_phase[robot]
            if phase == "parked" and now >= release_time:
                self.parked[robot] = False
                self._set_phase(robot, "spool", now)
                phase = "spool"

            if phase == "parked":
                self._publish_motor_speeds(robot, np.zeros(4))
            elif phase == "spool":
                self._spool_command(robot, now)
                if now - self.phase_start_time[robot] >= self.motor_spool_time:
                    self._set_phase(robot, "takeoff", now)
            elif phase == "takeoff":
                target = charging.takeoff_waypoint(robot, self.takeoff_height)
                self._service_command_to_target(
                    robot, target, positions, velocities, rotations, angular_velocities
                )
                if self._settled_at_target(
                    robot, target, positions, velocities, now, self.takeoff_settle_time
                ):
                    self._set_phase(robot, "deploy", now)
            elif phase == "deploy":
                target = self.scenario.initial_positions[robot]
                self._service_command_to_target(
                    robot, target, positions, velocities, rotations, angular_velocities
                )
                if self._settled_at_target(
                    robot, target, positions, velocities, now, self.service_settle_time
                ):
                    self._set_phase(robot, "ready", now)
            elif phase == "ready":
                self._service_command_to_target(
                    robot, self.scenario.initial_positions[robot],
                    positions, velocities, rotations, angular_velocities
                )

        ready = all(
            self.service_phase[i] == "ready"
            for i in np.flatnonzero(self.scenario.initial_active).tolist()
        )
        if ready:
            for robot in np.flatnonzero(self.scenario.initial_active).tolist():
                self.service_phase[robot] = "omrs"
            self.mission_start_time = now
            self.get_logger().info(
                "Initial team airborne and stabilized at the admissible mission "
                "initial condition; starting OMRS mission clock."
            )
        return ready

    def _start_due_join_takeoffs(self, scenario_time: float, now: float) -> None:
        for request in self.scenario.join_requests:
            robot = request.robot
            if scenario_time + 0.5 * self.scenario.dt < request.time:
                continue
            if self.manager.active[robot] or self.manager.staging[robot]:
                continue
            if self.service_phase[robot] == "parked":
                self.parked[robot] = False
                self._set_phase(robot, "join_spool", now)

    def _join_capture_ready(
        self, request, positions: np.ndarray, velocities: np.ndarray
    ) -> bool:
        """Return true once a joining robot is close enough for BLF handoff.

        Candidate neighbors are selected with the same feasibility criterion
        used by FormationManager._start_join.  We deliberately create
        prospective edges only after the joining vehicle has approached the
        formation under the service controller, so the established team is not
        asked to react to a prospective edge spanning several metres.
        """
        robot = request.robot
        candidates: list[tuple[float, int]] = []
        for j in np.flatnonzero(self.manager.active).tolist():
            desired_distance = float(
                np.linalg.norm(
                    self.manager.current_desired_positions[robot]
                    - self.manager.current_desired_positions[j]
                )
            )
            if not (
                self.scenario.collision_distance
                < desired_distance
                < self.scenario.sensing_distance
            ):
                continue
            distance = float(np.linalg.norm(positions[robot] - positions[j]))
            candidates.append((distance, j))
        if len(candidates) < request.num_edges:
            return False
        candidates.sort()
        selected = candidates[: request.num_edges]
        within_capture = all(
            self.scenario.collision_distance + self.join_capture_lower_margin <= distance
            <= self.scenario.sensing_distance + self.join_capture_margin
            for distance, _ in selected
        )
        # Scalar distance alone is not enough: a candidate can be within the
        # sensing annulus while lying on the wrong side of its neighbors.  Gate
        # the handoff using the actual relative-vector formation error.  The
        # tolerance is intentionally loose; prospective control still performs
        # the final approach.
        errors = []
        for _, j in selected:
            actual_relative = positions[robot] - positions[j]
            desired_relative = (
                self.manager.current_desired_positions[robot]
                - self.manager.current_desired_positions[j]
            )
            errors.append(float(np.linalg.norm(actual_relative - desired_relative)))
        close_to_edge_sets = max(errors, default=np.inf) <= self.join_capture_error_tolerance
        slow_enough = float(np.linalg.norm(velocities[robot])) <= self.join_capture_speed_tolerance
        return within_capture and close_to_edge_sets and slow_enough

    def _update_join_takeoff_service(
        self,
        now: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        rotations: np.ndarray,
        angular_velocities: np.ndarray,
    ) -> None:
        charging = self.scenario.charging_area
        assert charging is not None
        for request in self.scenario.join_requests:
            robot = request.robot
            phase = self.service_phase[robot]
            if phase == "join_spool":
                self._spool_command(robot, now)
                if now - self.phase_start_time[robot] >= self.motor_spool_time:
                    self._set_phase(robot, "join_takeoff", now)
            elif phase == "join_takeoff":
                target = charging.takeoff_waypoint(robot, self.takeoff_height)
                self._service_command_to_target(
                    robot, target, positions, velocities, rotations, angular_velocities
                )
                if self._settled_at_target(
                    robot, target, positions, velocities, now, self.takeoff_settle_time
                ):
                    self._set_phase(robot, "join_approach", now)
            elif phase == "join_approach":
                # Fix the translational gauge using the current active-team
                # centroid and approach the joining robot's corresponding
                # desired absolute representative. Prospective edges take over
                # only for the final portion of the rendezvous.
                target = self._desired_positions_for_visualization(positions)[robot]
                self._service_command_to_target(
                    robot, target, positions, velocities, rotations, angular_velocities
                )
                if self._join_capture_ready(request, positions, velocities):
                    self.join_ready[robot] = True
                    self._set_phase(robot, "join_ready", now)
                    self.get_logger().info(
                        f"Drone {robot}: rendezvous capture reached; enabling prospective edges "
                        f"(max edge error <= {self.join_capture_error_tolerance:.2f} m, "
                        f"speed <= {self.join_capture_speed_tolerance:.2f} m/s)."
                    )
            elif phase == "join_ready" and not self.manager.staging[robot]:
                target = self._desired_positions_for_visualization(positions)[robot]
                self._service_command_to_target(
                    robot, target, positions, velocities, rotations, angular_velocities
                )
            if self.manager.staging[robot] and phase in (
                "join_ready", "join_approach", "join_takeoff", "join_spool"
            ):
                self.join_ready[robot] = False
                self._set_phase(robot, "omrs_staging", now)
                states = [
                    state for state in self.manager.prospective.values()
                    if state.purpose == "join" and robot in state.edge
                ]
                details = []
                for state in sorted(states, key=lambda value: value.edge):
                    i, j = state.edge
                    delta = positions[i] - positions[j]
                    error = delta - state.desired_relative
                    details.append(
                        f"{state.edge}:rho={state.rho:.3f},|e|={np.linalg.norm(error):.3f},"
                        f"d={np.linalg.norm(delta):.3f}"
                    )
                self.get_logger().info(
                    f"Drone {robot}: candidate-only prospective control active; "
                    + "; ".join(details)
                )

    def _collision_pairs_for_handover_mode(
        self,
        positions: np.ndarray,
        controlled: np.ndarray,
        handover_mode: str,
    ) -> list[tuple[int, int]]:
        """Return collision-only pairs consistent with a join-control mode.

        Immediately before admission, a joining prospective edge also belongs
        to the symmetric all-pairs collision layer.  During a post-admission
        whole-controller homotopy, ``pre`` reconstructs that exact situation by
        keeping the newly established handover edges in the collision layer,
        whereas ``full`` uses the ordinary established-edge exclusion.
        """
        excluded = set(self.manager.established_edges)
        if handover_mode == "pre":
            excluded.difference_update(self.manager.join_handovers.keys())
        elif handover_mode != "full":
            raise ValueError("handover_mode must be 'pre' or 'full'.")
        excluded.update(
            edge
            for edge, state in self.manager.prospective.items()
            if state.purpose == "bridge"
        )
        participants = np.asarray(controlled, dtype=bool) | self.returning
        return self.high_level.nearby_collision_pairs(
            positions, participants, excluded_edges=excluded
        )

    @staticmethod
    def _blend_high_level_outputs(pre: dict, full: dict, alpha: float) -> dict:
        """Convexly blend two complete high-level controller evaluations.

        Blending the *complete* pre-switch and post-switch control laws makes
        the applied acceleration exactly continuous at admission.  This avoids
        relying on an interpolation internal to the backstepping coordinates,
        where saturation and velocity-error coupling make the effect of a
        partially activated edge less transparent.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out: dict = {}
        numeric_keys = (
            "u", "v_star", "v_tilde", "v_star_dot", "q", "q_edge",
            "q_collision", "distributed_damping", "local_damping",
            "edge_barrier", "collision_barrier", "barrier", "lyapunov",
        )
        for key in numeric_keys:
            a = pre[key]
            b = full[key]
            if np.isscalar(a) and np.isscalar(b):
                out[key] = (1.0 - alpha) * float(a) + alpha * float(b)
            else:
                out[key] = (1.0 - alpha) * np.asarray(a, dtype=float) + alpha * np.asarray(b, dtype=float)
        out["collision_pairs"] = sorted(
            set(tuple(edge) for edge in pre.get("collision_pairs", ()))
            | set(tuple(edge) for edge in full.get("collision_pairs", ()))
        )
        out["handover_alpha"] = alpha
        out["u_pre"] = np.asarray(pre["u"], dtype=float)
        out["u_full"] = np.asarray(full["u"], dtype=float)
        out["v_star_pre"] = np.asarray(pre["v_star"], dtype=float)
        out["v_star_full"] = np.asarray(full["v_star"], dtype=float)
        return out

    def _evaluate_formation_controller(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        controlled: np.ndarray,
    ) -> tuple[dict, dict | None, dict | None, float | None, float | None]:
        """Evaluate the OMRS controller with a bumpless join homotopy.

        Outside a join handover this is just the ordinary symmetric controller.
        During a handover we evaluate both the exact pre-switch candidate-only
        law and the full post-switch undirected law at the *same physical
        state*, then blend their complete outputs with a C2 time profile.
        """
        full_interactions = self.manager.controller_interactions("full")
        full_collision_pairs = self._collision_pairs_for_handover_mode(
            positions, controlled, "full"
        )
        full = self.high_level.control(
            positions, velocities, full_interactions, controlled, full_collision_pairs
        )
        if not self.manager.join_handovers:
            return full, None, None, None, None

        pre_interactions = self.manager.controller_interactions("pre")
        pre_collision_pairs = self._collision_pairs_for_handover_mode(
            positions, controlled, "pre"
        )
        pre = self.high_level.control(
            positions, velocities, pre_interactions, controlled, pre_collision_pairs
        )
        progresses = [
            self.manager.handover_progress(edge)
            for edge in sorted(self.manager.join_handovers)
        ]
        progresses = [value for value in progresses if value is not None]
        if not progresses:
            return full, None, None, None, None
        alpha = min(value[0] for value in progresses)
        alpha_dot = min(value[1] for value in progresses)
        blended = self._blend_high_level_outputs(pre, full, alpha)
        return blended, pre, full, float(alpha), float(alpha_dot)

    def _snapshot_collision_pairs(
        self,
        positions: np.ndarray,
        controlled: np.ndarray,
        established_edges: set[tuple[int, int]],
        interactions,
    ) -> list[tuple[int, int]]:
        excluded = set(established_edges)
        excluded.update(
            interaction.edge
            for interaction in interactions
            if interaction.prospective and interaction.responders is None
        )
        participants = np.asarray(controlled, dtype=bool) | self.returning
        return self.high_level.nearby_collision_pairs(
            positions, participants, excluded_edges=excluded
        )

    @staticmethod
    def _norm_rows(array: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(array, dtype=float), axis=1)

    def _append_transition_robot_rows(
        self,
        *,
        scenario_time: float,
        label: str,
        event_robot: int,
        edge: tuple[int, int] | None,
        alpha: float | None,
        high_level: dict,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        v_star = np.asarray(high_level["v_star"], dtype=float)
        v_star_dot = np.asarray(high_level["v_star_dot"], dtype=float)
        v_tilde = np.asarray(high_level["v_tilde"], dtype=float)
        q = np.asarray(high_level["q"], dtype=float)
        u = np.asarray(high_level["u"], dtype=float)
        for robot in range(self.num_robots):
            self.join_transition_log_rows.append(
                {
                    "scenario_time": float(scenario_time),
                    "label": label,
                    "join_robot": int(event_robot),
                    "edge": "" if edge is None else f"{edge[0]}-{edge[1]}",
                    "alpha": np.nan if alpha is None else float(alpha),
                    "robot": int(robot),
                    "position_x": float(positions[robot, 0]),
                    "position_y": float(positions[robot, 1]),
                    "position_z": float(positions[robot, 2]),
                    "speed": float(np.linalg.norm(velocities[robot])),
                    "v_star_norm": float(np.linalg.norm(v_star[robot])),
                    "v_star_dot_norm": float(np.linalg.norm(v_star_dot[robot])),
                    "v_tilde_norm": float(np.linalg.norm(v_tilde[robot])),
                    "q_norm": float(np.linalg.norm(q[robot])),
                    "u_raw_norm": float(np.linalg.norm(u[robot])),
                }
            )

    def _log_join_switch_snapshot(
        self,
        *,
        scenario_time: float,
        event,
        snapshot: dict[str, object],
        positions: np.ndarray,
        velocities: np.ndarray,
        post_high_level: dict,
    ) -> None:
        pre_interactions = snapshot["interactions"]
        pre_controlled = np.asarray(snapshot["controlled"], dtype=bool)
        pre_established = set(snapshot["established_edges"])
        pre_desired = np.asarray(snapshot["desired_positions"], dtype=float)
        post_desired = self.manager.current_desired_positions.copy()
        pre_collision_pairs = self._snapshot_collision_pairs(
            positions, pre_controlled, pre_established, pre_interactions
        )
        pre_high_level = self.high_level.control(
            positions, velocities, pre_interactions, pre_controlled, pre_collision_pairs
        )
        post_full_interactions = self.manager.controller_interactions("full")
        post_full_collision_pairs = self._collision_pairs_for_handover_mode(
            positions, self.manager.controlled.copy(), "full"
        )
        post_full_high_level = self.high_level.control(
            positions, velocities, post_full_interactions,
            self.manager.controlled.copy(), post_full_collision_pairs
        )

        retained = sorted(pre_established)
        max_rel_shift = 0.0
        for i, j in retained:
            before = pre_desired[i] - pre_desired[j]
            after = post_desired[i] - post_desired[j]
            max_rel_shift = max(max_rel_shift, float(np.linalg.norm(after - before)))
        old_members = np.asarray(snapshot["active"], dtype=bool)
        max_abs_shift = (
            float(np.max(np.linalg.norm(post_desired[old_members] - pre_desired[old_members], axis=1)))
            if np.any(old_members)
            else 0.0
        )

        self.get_logger().warning(
            "JOIN HANDOVER diagnostic at t=%.3f s, robot=%s, new_edges=%s: "
            "max retained desired-relative change=%.3e m; max old-member desired-position change=%.3e m"
            % (
                scenario_time,
                str(event.robot),
                list(event.edges_added),
                max_rel_shift,
                max_abs_shift,
            )
        )

        pre_u = self._norm_rows(np.asarray(pre_high_level["u"], dtype=float))
        blend_u = self._norm_rows(np.asarray(post_high_level["u"], dtype=float))
        full_u = self._norm_rows(np.asarray(post_full_high_level["u"], dtype=float))
        pre_vs = self._norm_rows(np.asarray(pre_high_level["v_star"], dtype=float))
        blend_vs = self._norm_rows(np.asarray(post_high_level["v_star"], dtype=float))
        full_vs = self._norm_rows(np.asarray(post_full_high_level["v_star"], dtype=float))
        for robot in range(self.num_robots):
            jump_full = float(np.linalg.norm(
                np.asarray(post_full_high_level["u"])[robot]
                - np.asarray(pre_high_level["u"])[robot]
            ))
            self.get_logger().warning(
                "  r%d switch: |u_pre|=%.4f |u_blend(alpha=0)|=%.4f "
                "|u_full|=%.4f |u_full-u_pre|=%.4f; "
                "|v*_pre|=%.4f |v*_blend|=%.4f |v*_full|=%.4f"
                % (
                    robot, pre_u[robot], blend_u[robot], full_u[robot], jump_full,
                    pre_vs[robot], blend_vs[robot], full_vs[robot],
                )
            )

        pre_by_edge = {interaction.edge: interaction for interaction in pre_interactions}
        post_by_edge = {interaction.edge: interaction for interaction in post_full_interactions}
        for edge in event.edges_added:
            i, j = edge
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            desired = post_desired[i] - post_desired[j]
            error_norm = float(np.linalg.norm(positions[i] - positions[j] - desired))
            pre_int = pre_by_edge.get(edge)
            post_int = post_by_edge.get(edge)
            pre_rho = float(pre_int.relaxation) if pre_int is not None else np.nan
            post_weights = post_int.response_weights if post_int is not None else None
            self.get_logger().warning(
                "  edge %s: d=%.4f m, |e_x|=%.4f m, rho_pre=%.6f, post_response_weights=%s"
                % (edge, distance, error_norm, pre_rho, str(post_weights))
            )

        self._append_transition_robot_rows(
            scenario_time=scenario_time,
            label="switch_pre",
            event_robot=int(event.robot),
            edge=None,
            alpha=0.0,
            high_level=pre_high_level,
            positions=positions,
            velocities=velocities,
        )
        self._append_transition_robot_rows(
            scenario_time=scenario_time,
            label="switch_post",
            event_robot=int(event.robot),
            edge=None,
            alpha=0.0,
            high_level=post_high_level,
            positions=positions,
            velocities=velocities,
        )
        self._append_transition_robot_rows(
            scenario_time=scenario_time,
            label="switch_full",
            event_robot=int(event.robot),
            edge=None,
            alpha=0.0,
            high_level=post_full_high_level,
            positions=positions,
            velocities=velocities,
        )
        self.active_join_diagnostic_until = max(
            self.active_join_diagnostic_until,
            scenario_time + self.join_handover_duration + 0.5,
        )
        self.last_switch_diagnostic_time = -np.inf

    def _log_join_handover_diagnostics(
        self,
        *,
        scenario_time: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        high_level: dict,
        pre_high_level: dict | None = None,
        full_high_level: dict | None = None,
        blend_alpha: float | None = None,
        blend_alpha_dot: float | None = None,
    ) -> None:
        if scenario_time > self.active_join_diagnostic_until:
            return
        if scenario_time - self.last_switch_diagnostic_time < self.switch_diagnostic_period - 1e-9:
            return
        self.last_switch_diagnostic_time = float(scenario_time)

        handover_edges = set(self.manager.join_handovers)
        retained_edges = sorted(set(self.manager.established_edges) - handover_edges)
        critical_retained = None
        if retained_edges:
            distances = [
                (float(np.linalg.norm(positions[i] - positions[j])), (i, j))
                for i, j in retained_edges
            ]
            critical_retained = max(distances, key=lambda item: item[0])

        for edge, state in sorted(self.manager.join_handovers.items()):
            progress = self.manager.handover_progress(edge)
            if progress is None:
                continue
            alpha, alpha_dot = progress
            if blend_alpha is not None:
                alpha = float(blend_alpha)
            if blend_alpha_dot is not None:
                alpha_dot = float(blend_alpha_dot)
            i, j = edge
            desired = self.manager.desired_relative_for_edge((i, j))
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            error_norm = float(np.linalg.norm(positions[i] - positions[j] - desired))
            rel_speed = float(np.linalg.norm(velocities[i] - velocities[j]))
            retained_text = ""
            if critical_retained is not None:
                d_ret, edge_ret = critical_retained
                retained_text = (
                    f" max_retained={edge_ret}:{d_ret:.3f}m "
                    f"(upper_margin={self.scenario.sensing_distance-d_ret:.3f}m)"
                )
            self.get_logger().info(
                "JOIN HOMOTOPY t=%.3f edge=%s candidate=%d alpha=%.3f alpha_dot=%.3f "
                "d=%.3f |e_x|=%.3f |v_rel|=%.3f%s"
                % (
                    scenario_time, edge, state.candidate, alpha, alpha_dot,
                    distance, error_norm, rel_speed, retained_text,
                )
            )

            robots_to_print = sorted(set(edge))
            for robot in robots_to_print:
                blend_u = float(np.linalg.norm(np.asarray(high_level["u"])[robot]))
                blend_vs = float(np.linalg.norm(np.asarray(high_level["v_star"])[robot]))
                if pre_high_level is not None and full_high_level is not None:
                    pre_u_vec = np.asarray(pre_high_level["u"])[robot]
                    full_u_vec = np.asarray(full_high_level["u"])[robot]
                    pre_u = float(np.linalg.norm(pre_u_vec))
                    full_u = float(np.linalg.norm(full_u_vec))
                    delta_u = float(np.linalg.norm(full_u_vec - pre_u_vec))
                    self.get_logger().info(
                        "  r%d: |u_pre|=%.4f |u_full|=%.4f |du|=%.4f "
                        "|u_blend|=%.4f |v*_blend|=%.4f"
                        % (robot, pre_u, full_u, delta_u, blend_u, blend_vs)
                    )
                else:
                    self.get_logger().info(
                        "  r%d: |v*|=%.4f |v*_dot|=%.4f |v_tilde|=%.4f "
                        "|q|=%.4f |u_raw|=%.4f"
                        % (
                            robot,
                            float(np.linalg.norm(np.asarray(high_level["v_star"])[robot])),
                            float(np.linalg.norm(np.asarray(high_level["v_star_dot"])[robot])),
                            float(np.linalg.norm(np.asarray(high_level["v_tilde"])[robot])),
                            float(np.linalg.norm(np.asarray(high_level["q"])[robot])),
                            blend_u,
                        )
                    )

            self._append_transition_robot_rows(
                scenario_time=scenario_time,
                label="handover_blend",
                event_robot=state.candidate,
                edge=edge,
                alpha=alpha,
                high_level=high_level,
                positions=positions,
                velocities=velocities,
            )
            if pre_high_level is not None:
                self._append_transition_robot_rows(
                    scenario_time=scenario_time,
                    label="handover_pre",
                    event_robot=state.candidate,
                    edge=edge,
                    alpha=alpha,
                    high_level=pre_high_level,
                    positions=positions,
                    velocities=velocities,
                )
            if full_high_level is not None:
                self._append_transition_robot_rows(
                    scenario_time=scenario_time,
                    label="handover_full",
                    event_robot=state.candidate,
                    edge=edge,
                    alpha=alpha,
                    high_level=full_high_level,
                    positions=positions,
                    velocities=velocities,
                )


    def _handle_new_switch_events(
        self, now: float, switch_snapshot: dict[str, object] | None = None
    ) -> None:
        new_events = self.manager.switch_events[self.handled_switch_events:]
        for event in new_events:
            self.get_logger().info(
                f"OMRS switch at t={event.time:.2f} s: {event.kind}; robot={event.robot}; "
                f"added={list(event.edges_added)}; removed={list(event.edges_removed)}"
            )
            if event.kind == "robot_join" and event.robot is not None:
                self._set_phase(event.robot, "omrs", now)
                if switch_snapshot is not None:
                    self._pending_join_switch_snapshot = {
                        **switch_snapshot,
                        "event": event,
                    }

            if event.kind == "bridge_activation":
                bridge_terms = []
                for edge in event.edges_added:
                    target = self.manager.desired_relative_for_edge(edge)
                    i, j = edge
                    node_implied = (
                        self.manager.current_desired_positions[i]
                        - self.manager.current_desired_positions[j]
                    )
                    bridge_terms.append(
                        "%s:d_target=%.3f,node_implied=%.3f"
                        % (
                            str(edge),
                            float(np.linalg.norm(target)),
                            float(np.linalg.norm(node_implied)),
                        )
                    )
                self.get_logger().info(
                    "BRIDGE ACTIVATION bumpless: node-level desired formation "
                    "unchanged; established bridge keeps prospective target. %s"
                    % "; ".join(bridge_terms)
                )

            if event.kind == "robot_leave" and event.robot is not None:
                if switch_snapshot is not None:
                    pre_map = {
                        interaction.edge: np.asarray(
                            interaction.desired_relative, dtype=float
                        )
                        for interaction in switch_snapshot["interactions"]
                        if not interaction.prospective
                    }
                    max_surviving_target_jump = 0.0
                    for edge in self.manager.established_edges:
                        if edge not in pre_map:
                            continue
                        after = self.manager.desired_relative_for_edge(edge)
                        max_surviving_target_jump = max(
                            max_surviving_target_jump,
                            float(np.linalg.norm(after - pre_map[edge])),
                        )
                    self.get_logger().info(
                        "DEPARTURE TARGET diagnostic: max surviving desired-relative "
                        "change across robot removal = %.3e m"
                        % max_surviving_target_jump
                    )
                self.returning[event.robot] = True
                self.parked[event.robot] = False
                self._set_phase(event.robot, "returning", now)
        self.handled_switch_events = len(self.manager.switch_events)

    def _update_landing_states(self, positions: np.ndarray, velocities: np.ndarray, now: float) -> None:
        charging = self.scenario.charging_area
        assert charging is not None
        for robot in np.flatnonzero(self.returning).tolist():
            if charging.landed(positions[robot], velocities[robot], robot):
                self.returning[robot] = False
                self.parked[robot] = True
                self._set_phase(robot, "parked", now)
                self._publish_motor_speeds(robot, np.zeros(4))
                self.get_logger().info(f"Drone {robot} reached charging pad; motors switched off.")

    def _safe_hover_or_off(self) -> None:
        for robot in range(self.num_robots):
            if self.parked[robot]:
                self._publish_motor_speeds(robot, np.zeros(4))
            else:
                self._publish_motor_speeds(robot, np.full(4, self.hover_speed))

    def _condition_formation_acceleration(
        self, robot: int, command: np.ndarray, dt: float
    ) -> np.ndarray:
        """Apply the Gazebo-only physical acceleration interface.

        The magnitude cap is retained as a physical envelope.  Acceleration
        slew limiting is optional and disabled by default.  In the fixed-team
        Gazebo logs a 1.2 m/s^3 slew limit created a destabilizing phase lag:
        for some robots the applied acceleration was almost opposite to the
        current backstepping command during the first braking transient.

        Topology-change smoothness is handled by the explicit prospective-edge
        admission / whole-controller homotopy, while the rotor model already
        supplies actuator dynamics.
        """
        command = np.asarray(command, dtype=float).copy()
        norm = float(np.linalg.norm(command))
        if norm > self.formation_maximum_acceleration:
            command *= self.formation_maximum_acceleration / norm

        if self.formation_acceleration_rate_limit <= 0.0:
            return command

        previous = self.last_flight_acceleration_command[robot]
        delta = command - previous
        delta_norm = float(np.linalg.norm(delta))
        max_delta = self.formation_acceleration_rate_limit * max(float(dt), 1e-6)
        if delta_norm > max_delta:
            command = previous + delta * (max_delta / delta_norm)
        return command

    @staticmethod
    def _limit_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).copy()
        norm = float(np.linalg.norm(vector))
        if norm > limit:
            vector *= limit / norm
        return vector

    def _completion_errors(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[float, float]:
        max_position_error = 0.0
        max_velocity_error = 0.0
        for i, j in self.manager.established_edges:
            desired = self.manager.desired_relative_for_edge((i, j))
            position_error = positions[i] - positions[j] - desired
            velocity_error = velocities[i] - velocities[j]
            max_position_error = max(max_position_error, float(np.linalg.norm(position_error)))
            max_velocity_error = max(max_velocity_error, float(np.linalg.norm(velocity_error)))
        return max_position_error, max_velocity_error

    def _all_membership_requests_completed(self) -> bool:
        joined = {
            event.robot for event in self.manager.switch_events
            if event.kind == "robot_join" and event.robot is not None
        }
        left = {
            event.robot for event in self.manager.switch_events
            if event.kind == "robot_leave" and event.robot is not None
        }
        return (
            all(request.robot in joined for request in self.scenario.join_requests)
            and all(request.robot in left for request in self.scenario.leave_requests)
        )

    def _completion_ready(
        self, scenario_time: float, positions: np.ndarray, velocities: np.ndarray, now: float
    ) -> bool:
        if not self._all_membership_requests_completed():
            self.completion_candidate_start = np.nan
            return False
        if self.manager.prospective or np.any(self.manager.staging) or np.any(self.returning):
            self.completion_candidate_start = np.nan
            return False

        position_error, velocity_error = self._completion_errors(positions, velocities)
        within = (
            position_error <= self.completion_position_tolerance
            and velocity_error <= self.completion_velocity_tolerance
        )
        if not within:
            self.completion_candidate_start = np.nan
            return False
        if not np.isfinite(self.completion_candidate_start):
            self.completion_candidate_start = now
            return False
        return now - self.completion_candidate_start >= self.completion_settle_time

    def _write_structured_logs(self, result, plot_scenario, run_dir) -> list[str]:
        paths = [str(path) for path in save_diagnostic_csvs(result, plot_scenario, run_dir)]

        transition_log_path = run_dir / "join_transition_log.csv"
        if self.join_transition_log_rows:
            fieldnames = list(self.join_transition_log_rows[0].keys())
            with transition_log_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.join_transition_log_rows)
            paths.append(str(transition_log_path))

        exception_log_path = run_dir / "controller_exceptions.csv"
        if self.exception_log_rows:
            fieldnames = list(self.exception_log_rows[0].keys())
            with exception_log_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.exception_log_rows)
            paths.append(str(exception_log_path))

        performance_log_path = run_dir / "runtime_performance.csv"
        performance_rows = list(self.runtime_performance_rows)
        if performance_rows:
            fieldnames = list(performance_rows[0].keys())
            with performance_log_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(performance_rows)
            paths.append(str(performance_log_path))
        return paths

    def _start_result_generation(
        self,
        scenario_time: float,
        *,
        status: str = "mission_complete",
        reason: str | None = None,
        synchronous: bool = False,
        include_animation: bool | None = None,
    ) -> None:
        if not self.save_results:
            self._results_done = True
            return
        if self._results_thread is not None:
            if synchronous and self._results_thread.is_alive():
                self._results_thread.join(timeout=120.0)
            return

        try:
            result, plot_scenario = self.recorder.build_result(self.manager.switch_events)
        except RuntimeError as exc:
            self.get_logger().warning(f"No result bundle written: {exc}")
            self._results_done = True
            return

        run_dir = create_run_directory(self.output_directory)
        self._result_directory = str(run_dir)
        position_error, velocity_error = (
            float(result.position_edge_error_norm[-1]),
            float(result.velocity_edge_error_norm[-1]),
        )
        summary = {
            "status": status,
            "reason": reason,
            "mission_time": float(scenario_time),
            "final_position_edge_error_norm": position_error,
            "final_velocity_edge_error_norm": velocity_error,
            "active_robots": np.flatnonzero(self.manager.active).tolist(),
            "parked_robots": np.flatnonzero(self.parked).tolist(),
            "established_edges": [list(edge) for edge in sorted(self.manager.established_edges)],
        }
        mission_performance = [
            row for row in self.runtime_performance_rows
            if row.get("phase") == "mission"
        ]
        if mission_performance:
            rtf = np.asarray(
                [float(row["local_real_time_factor"]) for row in mission_performance],
                dtype=float,
            )
            period_ratio = np.asarray(
                [float(row["sim_period_ratio"]) for row in mission_performance],
                dtype=float,
            )
            callback_ms = np.asarray(
                [float(row["callback_wall_ms"]) for row in mission_performance],
                dtype=float,
            )
            finite_rtf = rtf[np.isfinite(rtf)]
            finite_ratio = period_ratio[np.isfinite(period_ratio)]
            finite_callback = callback_ms[np.isfinite(callback_ms)]
            if finite_rtf.size:
                summary["median_local_real_time_factor"] = float(np.median(finite_rtf))
            if finite_ratio.size:
                summary["max_sim_period_ratio"] = float(np.max(finite_ratio))
                summary["control_interval_overrun_fraction"] = float(
                    np.mean(finite_ratio > 1.5)
                )
            if finite_callback.size:
                summary["max_callback_wall_ms"] = float(np.max(finite_callback))
        save_raw_history(result, run_dir, summary)
        structured_paths = self._write_structured_logs(result, plot_scenario, run_dir)
        render_animation = self.save_animation if include_animation is None else bool(include_animation)

        def worker() -> None:
            try:
                paths: list[str] = [
                    str(run_dir / "gazebo_history.npz"),
                    str(run_dir / "summary.json"),
                    *structured_paths,
                ]
                if render_animation:
                    from constrained_omrs_ros.gazebo_results import render_run_outputs

                    rendered = render_run_outputs(
                        result,
                        plot_scenario,
                        run_dir,
                        self.high_level.config.max_virtual_speed,
                        animation_fps=self.animation_fps,
                        animation_stride=self.animation_stride,
                    )
                else:
                    from constrained_omrs.plotting import save_plots
                    rendered = save_plots(
                        result, plot_scenario, run_dir, self.high_level.config.max_virtual_speed
                    )
                paths.extend(str(path) for path in rendered)
                self._result_paths = paths
            except Exception as exc:
                self._results_error = (
                    f"{type(exc).__name__}: {exc}. Raw Gazebo history and structured "
                    f"CSV logs were saved successfully in {run_dir}; only plot/animation "
                    "rendering failed."
                )
            finally:
                self._results_done = True

        if synchronous:
            worker()
            self.get_logger().info(
                f"Saved {status} run logs and plots in: {run_dir}"
            )
        else:
            self._results_thread = Thread(target=worker, name="omrs-result-writer", daemon=True)
            self._results_thread.start()
            self.get_logger().info(
                f"Mission data captured. Generating plots"
                f"{' and animation' if render_animation else ''} in {run_dir}."
            )

    def save_interrupted_results(self, reason: str = "user_interrupt") -> None:
        """Synchronously persist the current run before the ROS process exits.

        This is deliberately safe to call from ``main`` after Ctrl-C.  It saves
        the lossless history, human-readable robot/edge/exception CSV logs and
        all plots (plus the GIF when enabled) even if the controller never
        reached mission completion.
        """
        if self._result_directory is not None:
            if self._results_thread is not None and self._results_thread.is_alive():
                self._results_thread.join(timeout=120.0)
            return
        if self.mission_start_time is None:
            scenario_time = -1.0
        else:
            scenario_time = max(0.0, self._now_seconds() - self.mission_start_time)
        self.get_logger().warning(
            f"Run interrupted ({reason}); saving logs and plots before shutdown."
        )
        self._start_result_generation(
            scenario_time,
            status="interrupted",
            reason=reason,
            synchronous=True,
            include_animation=self.save_animation_on_interrupt,
        )

    def _begin_post_mission_recovery(self, now: float) -> None:
        """Release the OMRS graph and send every airborne robot back to charge.

        Mission metrics are snapshotted before this method is called, so the
        service-flight recovery cannot contaminate the formation-control plots.
        Once the scientific mission is complete the interaction graph no longer
        has a task-level role; clearing it also prevents BLF evaluations from
        being applied while vehicles intentionally separate toward their pads.
        Collision avoidance remains active through the all-returning service
        controller.
        """
        if self.post_mission_recovery_started:
            return
        self.post_mission_recovery_started = True

        airborne_team = np.flatnonzero(self.manager.active | self.manager.staging)
        self.manager.active[:] = False
        self.manager.staging[:] = False
        self.manager.established_edges.clear()
        self.manager.prospective.clear()
        self.manager.established_desired_overrides.clear()
        self.manager.join_handovers.clear()

        for robot in airborne_team.tolist():
            self.returning[robot] = True
            self.parked[robot] = False
            self.join_ready[robot] = False
            self._set_phase(robot, "returning", now)

        if airborne_team.size:
            self.get_logger().info(
                "Post-mission recovery started: returning drones %s to their charging pads."
                % airborne_team.tolist()
            )
        else:
            self.get_logger().info("Post-mission recovery: all drones are already parked.")

    def _declare_mission_complete(self, scenario_time: float, now: float) -> None:
        if self.mission_complete:
            return
        self.mission_complete = True
        self.completion_time = now
        position_error, velocity_error = self._completion_errors(
            np.vstack([state.position for state in self.states]),
            np.vstack([state.velocity_world for state in self.states]),
        )
        self.get_logger().info(
            "================ MISSION COMPLETE ================"
        )
        self.get_logger().info(
            f"OMRS mission completed at t={scenario_time:.2f} s; "
            f"max established-edge position error={position_error:.3f} m, "
            f"relative-speed error={velocity_error:.3f} m/s."
        )
        # Build the result snapshot *before* service recovery changes graph
        # membership.  The scientific plots therefore stop at mission
        # completion, while Gazebo can continue with a visually meaningful
        # return-to-charge sequence.
        self._start_result_generation(scenario_time)
        if self.return_to_charge_on_completion:
            self._begin_post_mission_recovery(now)

    def _poll_completion_output(self) -> None:
        if not self.mission_complete or not self._results_done:
            return

        if not self._results_logged:
            self._results_logged = True
            if self._results_error is not None:
                self.get_logger().error(
                    f"Mission completed, but result rendering failed: {self._results_error}"
                )
            elif self._result_directory is not None:
                self.get_logger().info(
                    f"Plots, animation and raw data are available in: {self._result_directory}"
                )
                self.get_logger().info(
                    "Main files: trajectories.png/.pdf, edge_distances.png/.pdf, "
                    "formation_errors.png/.pdf, formation_command_conditioning_error.png/.pdf, "
                    "acceleration_realization_error.png/.pdf, gazebo_mission.gif/.mp4 "
                    "and gazebo_history.npz."
                )

        if self.return_to_charge_on_completion and np.any(self.returning):
            return

        if self.stop_on_mission_complete and not self._shutdown_requested:
            self._shutdown_requested = True
            if self.return_to_charge_on_completion:
                self.get_logger().info(
                    "Mission results finished and all robots are parked; shutting down Gazebo."
                )
            else:
                self.get_logger().info(
                    "Result generation finished; shutting down the OMRS controller and Gazebo."
                )
            self._publish_all_off()
            self.timer.cancel()
            rclpy.shutdown()

    def _control_tick(self) -> None:
        """Timing wrapper around the simulation-time control callback.

        The flight controller itself uses ROS / Gazebo simulation time.  These
        wall-clock measurements are diagnostics only; they make it possible to
        distinguish a harmless low real-time factor from actual missed control
        updates caused by CPU / GUI contention.
        """
        wall_start = perf_counter()
        sim_start = self._now_seconds()
        previous_wall = self._last_control_wall_start
        previous_sim = self._last_control_sim_start
        try:
            self._control_tick_impl()
        finally:
            wall_duration = perf_counter() - wall_start
            wall_interval = (
                np.nan if previous_wall is None else max(wall_start - previous_wall, 0.0)
            )
            sim_dt = (
                np.nan if previous_sim is None else max(sim_start - previous_sim, 0.0)
            )
            local_rtf = (
                sim_dt / wall_interval
                if np.isfinite(sim_dt) and np.isfinite(wall_interval) and wall_interval > 1e-9
                else np.nan
            )
            scenario_time = (
                -1.0
                if self.mission_start_time is None
                else max(0.0, sim_start - self.mission_start_time)
            )
            self.runtime_performance_rows.append(
                {
                    "scenario_time": float(scenario_time),
                    "sim_dt": float(sim_dt),
                    "wall_interval_s": float(wall_interval),
                    "callback_wall_ms": float(1000.0 * wall_duration),
                    "local_real_time_factor": float(local_rtf),
                    "sim_period_ratio": float(sim_dt * self.control_rate_hz)
                    if np.isfinite(sim_dt)
                    else np.nan,
                    "phase": (
                        "recovery" if self.mission_complete
                        else "mission" if self.mission_start_time is not None
                        else "preflight"
                    ),
                }
            )
            self._last_control_wall_start = wall_start
            self._last_control_sim_start = sim_start

    def _control_tick_impl(self) -> None:
        now = self._now_seconds()
        if not self._all_states_ready():
            self._publish_all_off()
            if now - self.last_waiting_log_time > 2.0:
                missing = [i for i, state in enumerate(self.states) if not state.received]
                self.get_logger().info(f"Waiting for odometry from drones {missing}.")
                self.last_waiting_log_time = now
            return

        if self.preflight_start_time is None:
            self.preflight_start_time = now
            self.last_control_time = now
            self.get_logger().info(
                "All vehicle states received; beginning controlled spool-up, takeoff, "
                "and deployment of the initial team."
            )

        assert self.last_control_time is not None
        dt = now - self.last_control_time
        if dt <= 0.0:
            return
        positions, velocities, rotations, angular_velocities = self._state_arrays()

        stale = [
            i for i, state in enumerate(self.states)
            if now - (
                state.last_state_time
                if np.isfinite(state.last_state_time)
                else state.last_receive_time
            ) > self.odom_timeout
        ]
        if stale:
            self.get_logger().error(
                f"Stale Gazebo odometry for drones {stale}; commanding hover/off fallback."
            )
            self._safe_hover_or_off()
            self.last_control_time = now
            return

        try:
            if self.mission_start_time is None:
                self._run_initial_deployment(
                    now, positions, velocities, rotations, angular_velocities
                )
                self._log_preflight_diagnostics(now, positions, velocities)
                if now - self.last_marker_time >= 1.0 / self.marker_rate_hz:
                    self._publish_markers(positions)
                    self._publish_status(-1.0)
                    self.last_marker_time = now
                self.last_control_time = now
                return

            scenario_time = now - self.mission_start_time
            if not self.mission_complete:
                self._start_due_join_takeoffs(scenario_time, now)
                self._update_join_takeoff_service(
                    now, positions, velocities, rotations, angular_velocities
                )

                if self.manager.prospective:
                    self.manager.integrate_relaxations(dt, positions)
                switch_snapshot = {
                    "interactions": self.manager.controller_interactions(),
                    "controlled": self.manager.controlled.copy(),
                    "active": self.manager.active.copy(),
                    "established_edges": set(self.manager.established_edges),
                    "desired_positions": self.manager.current_desired_positions.copy(),
                }
                self.manager.process(
                    scenario_time, positions, velocities, join_ready=self.join_ready
                )
                self.manager.assert_established_graph_connected()
                self._handle_new_switch_events(now, switch_snapshot)

            # Returning robots (ordinary departures or post-mission recovery)
            # keep using the service-flight layer until each vehicle is safely
            # parked, even after the OMRS mission itself has completed.
            self._update_landing_states(positions, velocities, now)

            formation_controlled = self.manager.controlled.copy()

            # Evaluate a complete pre-switch / post-switch controller homotopy
            # during admission.  This is deliberately done at the level of the
            # finished backstepping acceleration command rather than by
            # partially activating the response incidence matrix inside the
            # nonlinear saturated controller.  At alpha=0 the command is
            # exactly the candidate-only pre-admission law; at alpha=1 it is
            # exactly the ordinary undirected post-admission law.
            high_level, handover_pre, handover_full, handover_alpha, handover_alpha_dot = (
                self._evaluate_formation_controller(
                    positions, velocities, formation_controlled
                )
            )
            self.last_collision_pairs = list(high_level.get("collision_pairs", ()))
            if self._pending_join_switch_snapshot is not None:
                event = self._pending_join_switch_snapshot["event"]
                self._log_join_switch_snapshot(
                    scenario_time=scenario_time,
                    event=event,
                    snapshot=self._pending_join_switch_snapshot,
                    positions=positions,
                    velocities=velocities,
                    post_high_level=high_level,
                )
                self._pending_join_switch_snapshot = None
            self._log_join_handover_diagnostics(
                scenario_time=scenario_time,
                positions=positions,
                velocities=velocities,
                high_level=high_level,
                pre_high_level=handover_pre,
                full_high_level=handover_full,
                blend_alpha=handover_alpha,
                blend_alpha_dot=handover_alpha_dot,
            )
            self._log_fixed_team_formation_diagnostics(
                scenario_time, positions, velocities
            )
            raw_acceleration_commands = np.asarray(high_level["u"], dtype=float).copy()
            applied_acceleration_commands = raw_acceleration_commands.copy()

            charging = self.scenario.charging_area
            assert charging is not None
            collision_q = np.asarray(high_level["q_collision"], dtype=float)
            for robot in np.flatnonzero(self.returning):
                return_command = charging.acceleration_command(
                    positions[robot], velocities[robot], int(robot)
                )
                # The returning robot is outside the graph but remains a
                # physical collision participant until it reaches its pad.
                return_command = return_command - collision_q[robot]
                applied_acceleration_commands[robot] = self._limit_vector_norm(
                    return_command, self.service_maximum_acceleration
                )

            # OMRS-controlled and returning vehicles use the geometric / rotor
            # stack. Joiners still in spool / vertical-takeoff phases were
            # already commanded by _update_join_takeoff_service above.  OMRS
            # commands pass through the Gazebo-only physical acceleration
            # envelope. Slew limiting is disabled by default; graph-switch
            # smoothness is handled by the explicit join homotopy.
            service_join = np.array(
                [
                    phase in ("join_spool", "join_takeoff", "join_approach", "join_ready")
                    for phase in self.service_phase
                ],
                dtype=bool,
            )
            flight_controlled = formation_controlled | self.returning
            for robot in range(self.num_robots):
                if service_join[robot]:
                    applied_acceleration_commands[robot] = self.last_flight_acceleration_command[robot]
                    continue
                if not flight_controlled[robot] or self.parked[robot]:
                    applied_acceleration_commands[robot] = 0.0
                    self.last_flight_acceleration_command[robot] = 0.0
                    self._publish_motor_speeds(robot, np.zeros(4))
                    continue
                if formation_controlled[robot] and not self.returning[robot]:
                    applied_acceleration_commands[robot] = self._condition_formation_acceleration(
                        robot, applied_acceleration_commands[robot], dt
                    )
                self._publish_acceleration_control(
                    robot, applied_acceleration_commands[robot], rotations,
                    angular_velocities, velocities
                )

            self.recorder.record(
                scenario_time=scenario_time,
                manager=self.manager,
                positions=positions,
                velocities=velocities,
                rotations=rotations,
                angular_velocities=angular_velocities,
                returning=self.returning,
                parked=self.parked,
                raw_acceleration_commands=raw_acceleration_commands,
                applied_acceleration_commands=applied_acceleration_commands,
                virtual_velocities=np.asarray(high_level["v_star"], dtype=float),
                lyapunov=float(high_level["lyapunov"]),
                attitude_errors=np.deg2rad(self.last_attitude_error_deg),
                commanded_thrusts=self.last_commanded_thrust,
                commanded_torques=self.last_commanded_torque,
            )

            if (
                not self.mission_complete
                and self._completion_ready(scenario_time, positions, velocities, now)
            ):
                self._declare_mission_complete(scenario_time, now)

            if now - self.last_marker_time >= 1.0 / self.marker_rate_hz:
                self._publish_markers(positions)
                self._publish_status(scenario_time)
                self.last_marker_time = now

            self._poll_completion_output()

        except (FloatingPointError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Controller exception: {exc}")
            if now - self.last_exception_snapshot_time >= self.switch_diagnostic_period - 1e-9:
                st = -1.0 if self.mission_start_time is None else max(0.0, now - self.mission_start_time)
                row: dict[str, object] = {
                    "time": float(st),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                for robot in range(self.num_robots):
                    row[f"r{robot}_px"] = float(positions[robot, 0])
                    row[f"r{robot}_py"] = float(positions[robot, 1])
                    row[f"r{robot}_pz"] = float(positions[robot, 2])
                    row[f"r{robot}_speed"] = float(np.linalg.norm(velocities[robot]))
                for edge in sorted(self.manager.established_edges):
                    i, j = edge
                    row[f"edge_{i}_{j}_distance"] = float(np.linalg.norm(positions[i] - positions[j]))
                self.exception_log_rows.append(row)
                self.last_exception_snapshot_time = now
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
        desired_marker.points = (
            [self._point(desired[i]) for i in np.flatnonzero(self.manager.controlled)]
            if self.mission_start_time is not None
            else []
        )
        markers.markers.append(desired_marker)

        established = Marker()
        established.header.frame_id = "world"
        established.header.stamp = stamp
        established.ns = "established_edges"
        established.id = 0
        established.type = Marker.LINE_LIST
        established.action = Marker.ADD
        established.scale.x = 0.070
        established.color.r = 0.03
        established.color.g = 0.40
        established.color.b = 0.08
        established.color.a = 0.95
        if self.mission_start_time is not None:
            dash_count = 12
            for i, j in sorted(self.manager.established_edges):
                p0 = np.asarray(positions[i], dtype=float)
                p1 = np.asarray(positions[j], dtype=float)
                for segment in range(0, dash_count, 2):
                    alpha0 = segment / dash_count
                    alpha1 = (segment + 1) / dash_count
                    established.points.extend(
                        [
                            self._point((1.0 - alpha0) * p0 + alpha0 * p1),
                            self._point((1.0 - alpha1) * p0 + alpha1 * p1),
                        ]
                    )
        markers.markers.append(established)

        prospective = Marker()
        prospective.header.frame_id = "world"
        prospective.header.stamp = stamp
        prospective.ns = "prospective_edges"
        prospective.id = 0
        prospective.type = Marker.LINE_LIST
        prospective.action = Marker.ADD
        prospective.scale.x = 0.055
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
        self._queue_native_gazebo_markers(positions)

    def _poll_native_gazebo_marker_result(self) -> None:
        result = self._gz_marker_result
        if result is None:
            return
        self._gz_marker_result = None
        ok, error = result
        if ok and not self._gz_marker_success_logged:
            self.get_logger().info(
                "Native Gazebo interaction-edge visualization is active "
                "(asynchronous, best-effort)."
            )
            self._gz_marker_success_logged = True
            self._gz_marker_failure_logged = False
        elif not ok and error is not None and not self._gz_marker_failure_logged:
            self.get_logger().warning(
                f"Native Gazebo marker update failed: {error}"
            )
            self._gz_marker_failure_logged = True

    def _queue_native_gazebo_markers(self, positions: np.ndarray) -> None:
        """Queue a best-effort native-Gazebo marker update off the control path."""
        self._poll_native_gazebo_marker_result()
        client = self.gz_edge_markers
        if (
            client is None
            or not client.available
            or self.mission_start_time is None
        ):
            return
        if self._gz_marker_thread is not None and self._gz_marker_thread.is_alive():
            # Visualization is deliberately lossy under load.  Keeping the
            # latest flight-control deadline is more important than drawing an
            # intermediate marker frame.
            return

        established_edges = tuple(sorted(self.manager.established_edges))
        prospective_edges = tuple(sorted(self.manager.prospective))
        positions_snapshot = np.asarray(positions, dtype=float).copy()

        def worker() -> None:
            ok = client.update_snapshot(
                established_edges=established_edges,
                prospective_edges=prospective_edges,
                positions=positions_snapshot,
            )
            self._gz_marker_result = (bool(ok), client.error)

        self._gz_marker_thread = Thread(
            target=worker,
            name="omrs-gazebo-marker-worker",
            daemon=True,
        )
        self._gz_marker_thread.start()

    def _publish_status(self, scenario_time: float) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "phase": (
                    "complete" if self.mission_complete
                    else "mission" if self.mission_start_time is not None
                    else "preflight"
                ),
                "time": None if scenario_time < 0.0 else round(float(scenario_time), 3),
                "service_phase": {str(i): phase for i, phase in enumerate(self.service_phase)},
                "active": np.flatnonzero(self.manager.active).tolist(),
                "staging": np.flatnonzero(self.manager.staging).tolist(),
                "returning": np.flatnonzero(self.returning).tolist(),
                "parked": np.flatnonzero(self.parked).tolist(),
                "established_edges": [list(edge) for edge in sorted(self.manager.established_edges)],
                "collision_only_pairs": [list(edge) for edge in self.last_collision_pairs],
                "mission_complete": self.mission_complete,
                "result_directory": self._result_directory,
                "prospective_edges": [
                    {
                        "edge": list(edge),
                        "purpose": state.purpose,
                        "rho": round(float(state.rho), 4),
                        "rho_dot": round(float(state.rho_dot), 4),
                        "relaxed_dmax": round(float(self.scenario.sensing_distance + state.rho), 4),
                        "protected_upper": round(
                            float(
                                self.scenario.sensing_distance
                                + state.rho
                                - self.scenario.reconfiguration.edge_activation_margin
                            ),
                            4,
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
    interrupted = False
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Persist a partial run on Ctrl-C / external shutdown.  Normal mission
        # completion already starts its own result generation and is not saved
        # a second time here.  Rendering is synchronous so the process cannot
        # exit before the plots / CSV logs are flushed to disk.
        if not node.mission_complete and node._result_directory is None:
            try:
                node.save_interrupted_results(
                    "keyboard_interrupt" if interrupted else "ros_shutdown"
                )
            except Exception as exc:
                node.get_logger().error(
                    f"Failed to save interrupted-run diagnostics: {type(exc).__name__}: {exc}"
                )
        if rclpy.ok():
            for robot in range(node.num_robots):
                node._publish_motor_speeds(robot, np.zeros(4))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
