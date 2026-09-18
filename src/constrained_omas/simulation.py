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

    for k, t in enumerate(time):
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
        flight_controlled = formation_controlled | returning
        edges = sorted(manager.established_edges)
        interactions = manager.controller_interactions()

        active_history[k] = active
        controlled_history[k] = formation_controlled
        flight_controlled_history[k] = flight_controlled
        returning_history[k] = returning
        parked_history[k] = parked
        desired_history[k] = manager.current_desired_positions
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
            output = controller.control(
                positions[k],
                velocities[k],
                interactions,
                formation_controlled,
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
        if scenario.charging_area is not None:
            for i in np.flatnonzero(returning):
                flight_command[i] = scenario.charging_area.acceleration_command(
                    positions[k, i],
                    velocities[k, i],
                    int(i),
                )
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
                desired_relative = (
                    manager.current_desired_positions[i]
                    - manager.current_desired_positions[j]
                )
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

        # Vehicles not in the team, not staging, and not returning are parked.
        next_flight_controlled = manager.controlled | returning
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
