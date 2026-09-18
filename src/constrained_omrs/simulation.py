from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .controller import SmoothBacksteppingController
from .graph import Edge, active_pairs
from .quadrotor import (
    GeometricQuadrotorController,
    QuadrotorConfig,
    QuadrotorRotorActuator,
    exp_so3,
    project_so3,
)
from .reconfiguration import FormationManager, SwitchEvent
from .scenario import Scenario


DynamicsModel = Literal["point_mass", "quadrotor", "quadrotor_rotor"]


@dataclass
class SimulationResult:
    time: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    controls: np.ndarray
    flight_acceleration_commands: np.ndarray
    realized_accelerations: np.ndarray
    virtual_velocities: np.ndarray
    lyapunov: np.ndarray
    active: np.ndarray
    controlled: np.ndarray
    flight_controlled: np.ndarray
    returning: np.ndarray
    parked: np.ndarray
    edge_history: list[list[Edge]]
    prospective_history: list[dict[Edge, float]]
    prospective_margin_history: list[dict[Edge, float]]
    desired_positions: np.ndarray
    min_pair_distance: np.ndarray
    max_established_edge_distance: np.ndarray
    position_edge_error_norm: np.ndarray
    velocity_edge_error_norm: np.ndarray
    switch_events: tuple[SwitchEvent, ...]
    dynamics_model: DynamicsModel = "point_mass"
    attitudes: np.ndarray | None = None
    desired_attitudes: np.ndarray | None = None
    angular_velocities: np.ndarray | None = None
    thrusts: np.ndarray | None = None
    commanded_thrusts: np.ndarray | None = None
    torques: np.ndarray | None = None
    commanded_torques: np.ndarray | None = None
    attitude_errors: np.ndarray | None = None
    desired_edge_history: list[dict[Edge, np.ndarray]] | None = None
    rotor_thrusts: np.ndarray | None = None
    rotor_thrust_commands: np.ndarray | None = None
    rotor_allocation_saturated: np.ndarray | None = None
    quadrotor_config: QuadrotorConfig | None = None

    @property
    def max_protected_edge_distance(self) -> np.ndarray:
        """Backward-compatible alias."""
        return self.max_established_edge_distance

    @property
    def join_event_times(self) -> np.ndarray:
        return np.asarray(
            [event.time for event in self.switch_events if event.kind == "robot_join"],
            dtype=float,
        )

    @property
    def acceleration_tracking_error(self) -> np.ndarray:
        """Formation-controller command minus realized translational acceleration."""
        return self.controls - self.realized_accelerations

    @property
    def flight_acceleration_tracking_error(self) -> np.ndarray:
        """Low-level flight command minus realized translational acceleration."""
        return self.flight_acceleration_commands - self.realized_accelerations

    @property
    def staging(self) -> np.ndarray:
        return self.controlled & ~self.active


def _pairwise_metrics(
    positions: np.ndarray,
    active: np.ndarray,
    established_edges: list[Edge],
) -> tuple[float, float]:
    pairs = active_pairs(active)
    min_distance = (
        min(float(np.linalg.norm(positions[i] - positions[j])) for i, j in pairs)
        if pairs
        else np.nan
    )
    max_edge_distance = (
        max(
            float(np.linalg.norm(positions[i] - positions[j]))
            for i, j in established_edges
        )
        if established_edges
        else np.nan
    )
    return min_distance, max_edge_distance


def _validate_simulation_inputs(
    scenario: Scenario,
    controller: SmoothBacksteppingController,
    dynamics: DynamicsModel,
) -> None:
    if controller.num_robots != scenario.num_robots:
        raise ValueError("Controller and scenario robot counts differ.")
    if controller.dimension != scenario.dimension:
        raise ValueError("Controller and scenario dimensions differ.")
    if dynamics in ("quadrotor", "quadrotor_rotor") and scenario.dimension != 3:
        raise ValueError("The full quadrotor models require a 3-D scenario.")


def simulate(
    scenario: Scenario,
    controller: SmoothBacksteppingController,
    *,
    dynamics: DynamicsModel = "point_mass",
    quadrotor_config: QuadrotorConfig | None = None,
) -> SimulationResult:
    """Simulate the OMRS at three levels of vehicle fidelity.

    ``point_mass`` is the paper-level double integrator. ``quadrotor`` uses a
    6-DoF rigid body with ideal collective-thrust/body-torque realization.
    ``quadrotor_rotor`` additionally allocates the desired wrench to four bounded
    rotors and propagates first-order rotor-thrust dynamics.

    When a 3-D scenario defines a charging area, initially inactive vehicles are
    parked there with motors off. A join request releases the corresponding
    drone from its pad into the prospective-edge controller. After a robot leaves
    the team, a separate return-to-charge flight controller brings it back to its
    assigned pad without re-entering the interaction graph.
    """
    if dynamics not in ("point_mass", "quadrotor", "quadrotor_rotor"):
        raise ValueError(
            "dynamics must be 'point_mass', 'quadrotor', or 'quadrotor_rotor'."
        )
    _validate_simulation_inputs(scenario, controller, dynamics)

    dt = scenario.dt
    time = np.arange(0.0, scenario.duration + 0.5 * dt, dt)
    steps = len(time)
    n = scenario.num_robots
    dim = scenario.dimension

    positions = np.zeros((steps, n, dim), dtype=float)
    velocities = np.zeros((steps, n, dim), dtype=float)
    controls = np.zeros((steps, n, dim), dtype=float)
    flight_acceleration_commands = np.zeros((steps, n, dim), dtype=float)
    realized_accelerations = np.zeros((steps, n, dim), dtype=float)
    virtual_velocities = np.zeros((steps, n, dim), dtype=float)
    lyapunov = np.full(steps, np.nan)
    active_history = np.zeros((steps, n), dtype=bool)
    controlled_history = np.zeros((steps, n), dtype=bool)
    flight_controlled_history = np.zeros((steps, n), dtype=bool)
    returning_history = np.zeros((steps, n), dtype=bool)
    parked_history = np.zeros((steps, n), dtype=bool)
    desired_history = np.zeros((steps, n, dim), dtype=float)
    desired_edge_history: list[dict[Edge, np.ndarray]] = []
    min_pair_distance = np.full(steps, np.nan)
    max_established_edge_distance = np.full(steps, np.nan)
    position_edge_error_norm = np.full(steps, np.nan)
    velocity_edge_error_norm = np.full(steps, np.nan)
    edge_history: list[list[Edge]] = []
    prospective_history: list[dict[Edge, float]] = []
    prospective_margin_history: list[dict[Edge, float]] = []

    attitudes: np.ndarray | None = None
    desired_attitudes: np.ndarray | None = None
    angular_velocities: np.ndarray | None = None
    thrusts: np.ndarray | None = None
    commanded_thrusts: np.ndarray | None = None
    torques: np.ndarray | None = None
    commanded_torques: np.ndarray | None = None
    attitude_errors: np.ndarray | None = None
    rotor_thrusts: np.ndarray | None = None
    rotor_thrust_commands: np.ndarray | None = None
    rotor_allocation_saturated: np.ndarray | None = None
    quad_controller: GeometricQuadrotorController | None = None
    rotor_actuator: QuadrotorRotorActuator | None = None

    if dynamics in ("quadrotor", "quadrotor_rotor"):
        quadrotor_config = quadrotor_config or QuadrotorConfig()
        quad_controller = GeometricQuadrotorController(quadrotor_config)
        attitudes = np.repeat(np.eye(3)[None, None, :, :], steps * n, axis=0).reshape(
            steps, n, 3, 3
        )
        desired_attitudes = attitudes.copy()
        angular_velocities = np.zeros((steps, n, 3), dtype=float)
        thrusts = np.zeros((steps, n), dtype=float)
        commanded_thrusts = np.zeros((steps, n), dtype=float)
        torques = np.zeros((steps, n, 3), dtype=float)
        commanded_torques = np.zeros((steps, n, 3), dtype=float)
        attitude_errors = np.zeros((steps, n), dtype=float)

        if dynamics == "quadrotor_rotor":
            rotor_actuator = QuadrotorRotorActuator(quadrotor_config)
            rotor_thrusts = np.zeros((steps, n, 4), dtype=float)
            rotor_thrust_commands = np.zeros((steps, n, 4), dtype=float)
            rotor_allocation_saturated = np.zeros((steps, n), dtype=bool)
            rotor_thrusts[0, scenario.initial_active, :] = (
                quadrotor_config.hover_rotor_thrust
            )

    positions[0] = scenario.initial_positions
    velocities[0] = scenario.initial_velocities
    velocities[0, ~scenario.initial_active] = 0.0

    manager = FormationManager(scenario)
    manager.assert_established_graph_connected()

    returning = np.zeros(n, dtype=bool)
    parked = np.zeros(n, dtype=bool)
    if scenario.charging_area is not None:
        parked[~scenario.initial_active] = True
        positions[0, parked] = scenario.charging_area.pad_positions[parked]

    previous_formation_controlled = manager.controlled.copy()
    handled_switch_events = 0

    # The paper-level point-mass model can start a prospective interaction
    # directly.  For the physical quadrotor models, a joining vehicle first
    # leaves its charging pad and approaches the active formation under a
    # separate bounded service controller.  Prospective BLF control takes over
    # only for the final rendezvous, matching the Gazebo validation pipeline.
    physical_service = (
        dynamics in ("quadrotor", "quadrotor_rotor")
        and scenario.charging_area is not None
    )
    join_ready = np.zeros(n, dtype=bool)
    service_phase = np.full(n, "parked", dtype=object)
    service_phase[scenario.initial_active] = "omrs"
    service_settle_start = np.full(n, np.nan, dtype=float)

    # Keep these values synchronized with the physical Gazebo validation.
    service_takeoff_height = 0.85
    service_maximum_speed = 0.45
    service_maximum_acceleration = 0.90
    service_position_gain = 0.45
    service_velocity_gain = 2.20
    service_position_tolerance = 0.15
    service_speed_tolerance = 0.12
    takeoff_settle_time = 0.80
    # Hand prospective control a meaningful part of the rendezvous instead of
    # waiting until the candidate is almost in its final formation slot.  The
    # lower-margin gate keeps the handoff away from the collision boundary,
    # while FormationManager initializes rho so every new edge is deliberately
    # inside the relaxed upper boundary.
    join_capture_margin = 0.50
    join_capture_lower_margin = 0.20
    join_capture_error_tolerance = 1.10
    join_capture_speed_tolerance = 0.45
    formation_maximum_acceleration = 0.85

    def limit_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).copy()
        norm = float(np.linalg.norm(vector))
        if norm > limit:
            vector *= limit / norm
        return vector

    def aligned_desired_position(robot: int, current_positions: np.ndarray) -> np.ndarray:
        active_ids = np.flatnonzero(manager.active)
        target = manager.current_desired_positions[robot].copy()
        if active_ids.size:
            translation = (
                np.mean(current_positions[active_ids], axis=0)
                - np.mean(manager.current_desired_positions[active_ids], axis=0)
            )
            target += translation
        return target

    def target_reached(
        robot: int, target: np.ndarray, current_positions: np.ndarray, current_velocities: np.ndarray
    ) -> bool:
        charging = scenario.charging_area
        assert charging is not None
        return charging.target_reached(
            current_positions[robot],
            current_velocities[robot],
            target,
            position_tolerance=service_position_tolerance,
            speed_tolerance=service_speed_tolerance,
        )

    def settled_at_target(
        robot: int,
        target: np.ndarray,
        current_positions: np.ndarray,
        current_velocities: np.ndarray,
        current_time: float,
        required_time: float,
    ) -> bool:
        if not target_reached(robot, target, current_positions, current_velocities):
            service_settle_start[robot] = np.nan
            return False
        if not np.isfinite(service_settle_start[robot]):
            service_settle_start[robot] = current_time
            return required_time <= 0.0
        return current_time - service_settle_start[robot] >= required_time

    def service_collision_correction(
        robot: int, current_positions: np.ndarray, current_parked: np.ndarray
    ) -> np.ndarray:
        participants = ~current_parked
        pairs = controller.nearby_collision_pairs(current_positions, participants)
        if not pairs:
            return np.zeros(dim, dtype=float)
        _, _, q_collision, _ = controller.collision_quantities(current_positions, pairs)
        return -np.asarray(q_collision[robot], dtype=float)

    def service_command_to_target(
        robot: int,
        target: np.ndarray,
        current_positions: np.ndarray,
        current_velocities: np.ndarray,
        current_parked: np.ndarray,
    ) -> np.ndarray:
        charging = scenario.charging_area
        assert charging is not None
        command = charging.point_acceleration_command(
            current_positions[robot],
            current_velocities[robot],
            target,
            maximum_speed=service_maximum_speed,
            maximum_acceleration=service_maximum_acceleration,
            position_gain=service_position_gain,
            velocity_gain=service_velocity_gain,
        )
        command += service_collision_correction(robot, current_positions, current_parked)
        return limit_vector_norm(command, service_maximum_acceleration)

    def join_capture_ready(
        request, current_positions: np.ndarray, current_velocities: np.ndarray
    ) -> bool:
        robot = request.robot
        candidates: list[tuple[float, int]] = []
        for j in np.flatnonzero(manager.active).tolist():
            desired_distance = float(
                np.linalg.norm(
                    manager.current_desired_positions[robot]
                    - manager.current_desired_positions[j]
                )
            )
            if not (
                scenario.collision_distance
                < desired_distance
                < scenario.sensing_distance
            ):
                continue
            distance = float(np.linalg.norm(current_positions[robot] - current_positions[j]))
            candidates.append((distance, j))
        if len(candidates) < request.num_edges:
            return False
        candidates.sort()
        selected = candidates[: request.num_edges]
        within_capture = all(
            scenario.collision_distance + join_capture_lower_margin <= distance
            <= scenario.sensing_distance + join_capture_margin
            for distance, _ in selected
        )
        errors = []
        for _, j in selected:
            actual_relative = current_positions[robot] - current_positions[j]
            desired_relative = (
                manager.current_desired_positions[robot]
                - manager.current_desired_positions[j]
            )
            errors.append(float(np.linalg.norm(actual_relative - desired_relative)))
        close_to_edge_sets = (
            max(errors, default=np.inf) <= join_capture_error_tolerance
        )
        slow_enough = (
            float(np.linalg.norm(current_velocities[robot]))
            <= join_capture_speed_tolerance
        )
        return within_capture and close_to_edge_sets and slow_enough

    def collision_pairs_for_handover_mode(
        current_positions: np.ndarray, controlled: np.ndarray, mode: str
    ) -> list[Edge]:
        excluded = set(manager.established_edges)
        if mode == "pre":
            excluded.difference_update(manager.join_handovers.keys())
        elif mode != "full":
            raise ValueError("mode must be 'pre' or 'full'.")
        excluded.update(
            edge
            for edge, state in manager.prospective.items()
            if state.purpose == "bridge"
        )
        participants = np.asarray(controlled, dtype=bool) | returning
        return controller.nearby_collision_pairs(
            current_positions, participants, excluded_edges=excluded
        )

    def blend_high_level_outputs(pre: dict, full: dict, alpha: float) -> dict:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out = dict(full)
        for key in (
            "u", "v_star", "v_tilde", "v_star_dot", "q", "q_edge",
            "q_collision", "distributed_damping", "local_damping",
            "edge_barrier", "collision_barrier", "barrier", "lyapunov",
        ):
            a = pre[key]
            b = full[key]
            if np.isscalar(a) and np.isscalar(b):
                out[key] = (1.0 - alpha) * float(a) + alpha * float(b)
            else:
                out[key] = (
                    (1.0 - alpha) * np.asarray(a, dtype=float)
                    + alpha * np.asarray(b, dtype=float)
                )
        out["collision_pairs"] = sorted(
            set(tuple(edge) for edge in pre.get("collision_pairs", ()))
            | set(tuple(edge) for edge in full.get("collision_pairs", ()))
        )
        return out

    def evaluate_formation_controller(
        current_positions: np.ndarray,
        current_velocities: np.ndarray,
        controlled: np.ndarray,
    ) -> dict:
        full_interactions = manager.controller_interactions("full")
        full_collision_pairs = collision_pairs_for_handover_mode(
            current_positions, controlled, "full"
        )
        full = controller.control(
            current_positions,
            current_velocities,
            full_interactions,
            controlled,
            full_collision_pairs,
        )
        if not manager.join_handovers:
            return full
        pre_interactions = manager.controller_interactions("pre")
        pre_collision_pairs = collision_pairs_for_handover_mode(
            current_positions, controlled, "pre"
        )
        pre = controller.control(
            current_positions,
            current_velocities,
            pre_interactions,
            controlled,
            pre_collision_pairs,
        )
        progresses = [
            manager.handover_progress(edge)
            for edge in sorted(manager.join_handovers)
        ]
        progresses = [value for value in progresses if value is not None]
        if not progresses:
            return full
        alpha = min(value[0] for value in progresses)
        return blend_high_level_outputs(pre, full, alpha)

    for k, t in enumerate(time):
        service_commands = np.zeros((n, dim), dtype=float)
        service_join = np.zeros(n, dtype=bool)

        if physical_service:
            charging = scenario.charging_area
            assert charging is not None
            # Release due joining vehicles from their pads, but keep them outside
            # the OMRS until the service rendezvous gate is satisfied.
            for request in scenario.join_requests:
                robot = request.robot
                if float(t) + 0.5 * dt < request.time:
                    continue
                if manager.active[robot] or manager.staging[robot]:
                    continue
                if service_phase[robot] == "parked":
                    parked[robot] = False
                    service_phase[robot] = "join_takeoff"
                    service_settle_start[robot] = np.nan
                    if dynamics == "quadrotor_rotor":
                        assert rotor_thrusts is not None
                        assert quadrotor_config is not None
                        rotor_thrusts[k, robot, :] = quadrotor_config.hover_rotor_thrust

            for request in scenario.join_requests:
                robot = request.robot
                phase = service_phase[robot]
                if phase == "join_takeoff":
                    service_join[robot] = True
                    target = charging.takeoff_waypoint(robot, service_takeoff_height)
                    service_commands[robot] = service_command_to_target(
                        robot, target, positions[k], velocities[k], parked
                    )
                    if settled_at_target(
                        robot, target, positions[k], velocities[k], float(t), takeoff_settle_time
                    ):
                        service_phase[robot] = "join_approach"
                        service_settle_start[robot] = np.nan
                elif phase == "join_approach":
                    service_join[robot] = True
                    target = aligned_desired_position(robot, positions[k])
                    service_commands[robot] = service_command_to_target(
                        robot, target, positions[k], velocities[k], parked
                    )
                    if join_capture_ready(request, positions[k], velocities[k]):
                        join_ready[robot] = True
                        service_phase[robot] = "join_ready"
                elif phase == "join_ready" and not manager.staging[robot]:
                    service_join[robot] = True
                    target = aligned_desired_position(robot, positions[k])
                    service_commands[robot] = service_command_to_target(
                        robot, target, positions[k], velocities[k], parked
                    )

            manager.process(
                float(t), positions[k], velocities[k], join_ready=join_ready
            )
            for request in scenario.join_requests:
                robot = request.robot
                if manager.staging[robot] and service_phase[robot] in (
                    "join_takeoff", "join_approach", "join_ready"
                ):
                    join_ready[robot] = False
                    service_phase[robot] = "omrs_staging"
                    service_join[robot] = False
                    service_commands[robot] = 0.0
                elif manager.active[robot] and service_phase[robot] == "omrs_staging":
                    service_phase[robot] = "omrs"
        else:
            manager.process(float(t), positions[k], velocities[k])

        manager.assert_established_graph_connected()

        # Membership switches are intentionally distinct from service flights.
        # A robot that has left the graph immediately stops contributing to the
        # OMRS controller, but it remains a physical vehicle and returns home.
        new_events = manager.switch_events[handled_switch_events:]
        for event in new_events:
            if event.kind == "robot_leave" and event.robot is not None:
                if scenario.charging_area is not None:
                    returning[event.robot] = True
                    parked[event.robot] = False
        handled_switch_events = len(manager.switch_events)

        active = manager.active.copy()
        formation_controlled = manager.controlled.copy()
        newly_controlled = formation_controlled & ~previous_formation_controlled
        if scenario.charging_area is not None:
            parked[newly_controlled] = False

        # On a real platform the vehicle can spool up while still supported by
        # the charging pad. We initialize a newly released rotor-state model at
        # hover thrust to represent that pre-takeoff spool-up without introducing
        # a separate ground-contact model.
        if dynamics == "quadrotor_rotor" and np.any(newly_controlled):
            assert rotor_thrusts is not None
            assert quadrotor_config is not None
            rotor_thrusts[k, newly_controlled, :] = quadrotor_config.hover_rotor_thrust

        previous_formation_controlled = formation_controlled.copy()
        flight_controlled = formation_controlled | returning | service_join
        edges = sorted(manager.established_edges)

        active_history[k] = active
        controlled_history[k] = formation_controlled
        flight_controlled_history[k] = flight_controlled
        returning_history[k] = returning
        parked_history[k] = parked
        desired_history[k] = manager.current_desired_positions
        desired_edge_history.append(
            {
                edge: manager.desired_relative_for_edge(edge).copy()
                for edge in manager.established_edges
            }
            | {
                edge: state.desired_relative.copy()
                for edge, state in manager.prospective.items()
            }
        )
        edge_history.append(edges)
        prospective_history.append(
            {edge: float(state.rho) for edge, state in manager.prospective.items()}
        )
        prospective_margin_history.append(
            {
                edge: float(
                    scenario.sensing_distance
                    + state.rho
                    - np.linalg.norm(positions[k, edge[0]] - positions[k, edge[1]])
                )
                for edge, state in manager.prospective.items()
            }
        )

        try:
            output = evaluate_formation_controller(
                positions[k], velocities[k], formation_controlled
            )
        except FloatingPointError as exc:
            raise FloatingPointError(
                f"{exc} (t={t:.3f} s, established={edges}, "
                f"prospective={list(manager.prospective)})"
            ) from exc
        u = np.asarray(output["u"], dtype=float)
        v_star = np.asarray(output["v_star"], dtype=float)
        controls[k] = u
        virtual_velocities[k] = v_star
        lyapunov[k] = float(output["lyapunov"])

        flight_command = u.copy()
        if physical_service:
            for i in np.flatnonzero(formation_controlled):
                flight_command[i] = limit_vector_norm(
                    flight_command[i], formation_maximum_acceleration
                )

        if scenario.charging_area is not None:
            collision_q = np.asarray(output["q_collision"], dtype=float)
            for i in np.flatnonzero(returning):
                return_command = scenario.charging_area.acceleration_command(
                    positions[k, i],
                    velocities[k, i],
                    int(i),
                )
                # A departed vehicle is outside the interaction graph but is
                # still a physical collision participant while returning home.
                return_command -= collision_q[i]
                flight_command[i] = limit_vector_norm(
                    return_command, service_maximum_acceleration
                )

        for i in np.flatnonzero(service_join):
            flight_command[i] = service_commands[i]

        flight_acceleration_commands[k] = flight_command

        if dynamics == "point_mass":
            realized_accelerations[k, flight_controlled] = flight_command[flight_controlled]
        else:
            assert quad_controller is not None
            assert attitudes is not None
            assert desired_attitudes is not None
            assert angular_velocities is not None
            assert thrusts is not None
            assert commanded_thrusts is not None
            assert torques is not None
            assert commanded_torques is not None
            assert attitude_errors is not None
            assert quadrotor_config is not None

            for i in np.flatnonzero(flight_controlled):
                low_level = quad_controller.evaluate(
                    attitudes[k, i],
                    angular_velocities[k, i],
                    velocities[k, i],
                    flight_command[i],
                )
                desired_attitudes[k, i] = low_level.desired_rotation
                commanded_thrusts[k, i] = low_level.thrust
                commanded_torques[k, i] = low_level.torque
                attitude_errors[k, i] = low_level.attitude_error

                if dynamics == "quadrotor":
                    thrusts[k, i] = low_level.thrust
                    torques[k, i] = low_level.torque
                    realized_accelerations[k, i] = low_level.realized_acceleration
                else:
                    assert rotor_actuator is not None
                    assert rotor_thrusts is not None
                    assert rotor_thrust_commands is not None
                    assert rotor_allocation_saturated is not None
                    allocation = rotor_actuator.allocate(
                        low_level.thrust,
                        low_level.torque,
                    )
                    rotor_thrust_commands[k, i] = allocation.commands
                    rotor_allocation_saturated[k, i] = allocation.saturated
                    actual_wrench = rotor_actuator.wrench(rotor_thrusts[k, i])
                    thrusts[k, i] = actual_wrench[0]
                    torques[k, i] = actual_wrench[1:]
                    realized_accelerations[k, i] = quad_controller.translational_acceleration(
                        attitudes[k, i],
                        velocities[k, i],
                        actual_wrench[0],
                    )

        min_pair_distance[k], max_established_edge_distance[k] = _pairwise_metrics(
            positions[k], active, edges
        )

        if edges:
            position_errors = []
            velocity_errors = []
            for i, j in edges:
                desired_relative = manager.desired_relative_for_edge((i, j))
                position_errors.append(
                    (positions[k, i] - positions[k, j]) - desired_relative
                )
                velocity_errors.append(velocities[k, i] - velocities[k, j])
            position_edge_error_norm[k] = float(np.linalg.norm(position_errors))
            velocity_edge_error_norm[k] = float(np.linalg.norm(velocity_errors))

        if k == steps - 1:
            break

        positions[k + 1] = positions[k]
        velocities[k + 1] = 0.0

        if dynamics == "point_mass":
            positions[k + 1, flight_controlled] = (
                positions[k, flight_controlled]
                + dt * velocities[k, flight_controlled]
                + 0.5 * dt**2 * flight_command[flight_controlled]
            )
            velocities[k + 1, flight_controlled] = (
                velocities[k, flight_controlled]
                + dt * flight_command[flight_controlled]
            )
        else:
            assert quad_controller is not None
            assert attitudes is not None
            assert angular_velocities is not None
            assert torques is not None
            attitudes[k + 1] = attitudes[k]
            angular_velocities[k + 1] = 0.0

            if dynamics == "quadrotor_rotor":
                assert rotor_actuator is not None
                assert rotor_thrusts is not None
                assert rotor_thrust_commands is not None
                rotor_thrusts[k + 1] = rotor_thrusts[k]

            for i in np.flatnonzero(flight_controlled):
                acceleration = realized_accelerations[k, i]
                positions[k + 1, i] = (
                    positions[k, i]
                    + dt * velocities[k, i]
                    + 0.5 * dt**2 * acceleration
                )
                velocities[k + 1, i] = velocities[k, i] + dt * acceleration

                omega = angular_velocities[k, i]
                omega_dot = quad_controller.angular_acceleration(omega, torques[k, i])
                omega_next = omega + dt * omega_dot
                omega_mid = 0.5 * (omega + omega_next)
                attitudes[k + 1, i] = project_so3(
                    attitudes[k, i] @ exp_so3(dt * omega_mid)
                )
                angular_velocities[k + 1, i] = omega_next

                if dynamics == "quadrotor_rotor":
                    rotor_thrusts[k + 1, i] = rotor_actuator.integrate(
                        rotor_thrusts[k, i],
                        rotor_thrust_commands[k, i],
                        dt,
                    )

        manager.integrate_relaxations(dt, positions[k + 1])

        # Finish a return flight only after the vehicle has physically reached
        # its pad. The landed vehicle is then held by the charging station and
        # its motors are switched off; no contact model is needed for this event.
        if scenario.charging_area is not None:
            for i in np.flatnonzero(returning).tolist():
                if scenario.charging_area.landed(
                    positions[k + 1, i],
                    velocities[k + 1, i],
                    i,
                ):
                    positions[k + 1, i] = scenario.charging_area.pad(i)
                    velocities[k + 1, i] = 0.0
                    returning[i] = False
                    parked[i] = True
                    if dynamics in ("quadrotor", "quadrotor_rotor"):
                        assert attitudes is not None
                        assert angular_velocities is not None
                        attitudes[k + 1, i] = np.eye(3)
                        angular_velocities[k + 1, i] = 0.0
                    if dynamics == "quadrotor_rotor":
                        assert rotor_thrusts is not None
                        rotor_thrusts[k + 1, i] = 0.0

        # Vehicles not in the team, not in a service-flight join phase, and not
        # returning are held at rest on their charging pads.
        next_service_join = np.array(
            [
                phase in ("join_takeoff", "join_approach", "join_ready")
                for phase in service_phase
            ],
            dtype=bool,
        )
        next_flight_controlled = manager.controlled | returning | next_service_join
        velocities[k + 1, ~next_flight_controlled] = 0.0
        if dynamics in ("quadrotor", "quadrotor_rotor"):
            assert angular_velocities is not None
            angular_velocities[k + 1, ~next_flight_controlled] = 0.0

    return SimulationResult(
        time=time,
        positions=positions,
        velocities=velocities,
        controls=controls,
        flight_acceleration_commands=flight_acceleration_commands,
        realized_accelerations=realized_accelerations,
        virtual_velocities=virtual_velocities,
        lyapunov=lyapunov,
        active=active_history,
        controlled=controlled_history,
        flight_controlled=flight_controlled_history,
        returning=returning_history,
        parked=parked_history,
        edge_history=edge_history,
        prospective_history=prospective_history,
        prospective_margin_history=prospective_margin_history,
        desired_positions=desired_history,
        desired_edge_history=desired_edge_history,
        min_pair_distance=min_pair_distance,
        max_established_edge_distance=max_established_edge_distance,
        position_edge_error_norm=position_edge_error_norm,
        velocity_edge_error_norm=velocity_edge_error_norm,
        switch_events=tuple(manager.switch_events),
        dynamics_model=dynamics,
        attitudes=attitudes,
        desired_attitudes=desired_attitudes,
        angular_velocities=angular_velocities,
        thrusts=thrusts,
        commanded_thrusts=commanded_thrusts,
        torques=torques,
        commanded_torques=commanded_torques,
        attitude_errors=attitude_errors,
        rotor_thrusts=rotor_thrusts,
        rotor_thrust_commands=rotor_thrust_commands,
        rotor_allocation_saturated=rotor_allocation_saturated,
        quadrotor_config=quadrotor_config if dynamics in ("quadrotor", "quadrotor_rotor") else None,
    )
