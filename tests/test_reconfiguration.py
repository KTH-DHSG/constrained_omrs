from dataclasses import replace

import numpy as np
import pytest

from constrained_omrs import (
    ControllerConfig,
    FormationManager,
    SmoothBacksteppingController,
    default_open_formation_scenario,
)


def test_join_request_creates_relaxed_prospective_edge_with_interior_margin() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    manager = FormationManager(scenario)

    t = scenario.join_requests[0].time
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    manager.process(t, positions, velocities)

    assert manager.staging[4]
    assert not manager.active[4]
    assert len(manager.prospective) == 1
    state = next(iter(manager.prospective.values()))
    assert state.purpose == "join"

    i, j = state.edge
    distance = np.linalg.norm(positions[i] - positions[j])
    cfg = scenario.reconfiguration
    expected = max(
        0.0,
        distance
        - scenario.sensing_distance
        + cfg.edge_activation_margin
        + cfg.rho_initial_margin,
    )
    assert state.rho == pytest.approx(expected)
    protected_margin = (
        scenario.sensing_distance
        + state.rho
        - distance
        - cfg.edge_activation_margin
    )
    assert protected_margin >= cfg.rho_initial_margin - 1e-12


def test_relaxation_rate_uses_linear_nominal_contraction_and_projection() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    manager.process(scenario.join_requests[0].time, positions, velocities)

    state = next(iter(manager.prospective.values()))
    i, j = state.edge
    velocities[i] = np.array([0.25, -0.04])
    velocities[j] = np.array([-0.10, 0.02])
    manager.update_relaxation_rates(positions, velocities)

    delta = positions[i] - positions[j]
    distance = np.linalg.norm(delta)
    distance_rate = delta @ (velocities[i] - velocities[j]) / distance
    cfg = scenario.reconfiguration
    protected_margin = (
        scenario.sensing_distance
        + state.rho
        - distance
        - cfg.edge_activation_margin
    )
    expected = max(
        -cfg.rho_contraction_rate,
        distance_rate - cfg.lambda_rho * protected_margin,
    )
    assert state.rho_dot == pytest.approx(expected)


def test_linear_relaxation_reaches_exact_zero_when_projection_is_inactive() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    manager.process(scenario.join_requests[0].time, positions, velocities)

    state = next(iter(manager.prospective.values()))
    # Put the pair well inside the nominal upper boundary so the safety
    # projection is inactive while rho contracts at a constant rate.
    i, j = state.edge
    positions[i] = positions[j] + np.array([1.0, 0.0])
    state.rho = 0.037
    manager.update_relaxation_rates(positions, velocities)
    assert state.rho_dot == pytest.approx(-scenario.reconfiguration.rho_contraction_rate)

    dt = 0.01
    for _ in range(1000):
        manager.integrate_relaxations(dt, positions)
        manager.update_relaxation_rates(positions, velocities)
        if state.rho == 0.0:
            break
    assert state.rho == 0.0


def test_join_ready_gate_delays_prospective_edge_assignment() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    ready = np.ones(scenario.num_robots, dtype=bool)
    robot = scenario.join_requests[0].robot
    ready[robot] = False

    manager.process(scenario.join_requests[0].time, positions, velocities, join_ready=ready)
    assert not manager.staging[robot]
    assert not manager.prospective

    ready[robot] = True
    manager.process(
        scenario.join_requests[0].time + scenario.dt,
        positions,
        velocities,
        join_ready=ready,
    )
    assert manager.staging[robot]
    assert manager.prospective


def test_join_prospective_edges_act_only_on_candidate_before_admission() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    robot = scenario.join_requests[0].robot
    manager.process(scenario.join_requests[0].time, positions, velocities)

    interactions = manager.controller_interactions()
    join_interactions = [interaction for interaction in interactions if interaction.prospective]
    assert join_interactions
    assert all(interaction.responders == (robot,) for interaction in join_interactions)

    controller = SmoothBacksteppingController(
        num_robots=scenario.num_robots,
        dimension=scenario.dimension,
        collision_distance=scenario.collision_distance,
        sensing_distance=scenario.sensing_distance,
        config=ControllerConfig(k1=0.5, k2=0.4, k3=1.2, max_virtual_speed=0.5),
    )
    # Isolate the prospective interactions from the established graph.  Their
    # gradient and pairwise damping must have zero contribution on every
    # existing-team endpoint.
    controlled = manager.controlled.copy()
    output = controller.control(
        positions,
        velocities,
        join_interactions,
        controlled,
        collision_pairs=(),
    )
    q = np.asarray(output["q_edge"])
    u = np.asarray(output["u"])
    existing = np.flatnonzero(manager.active)
    np.testing.assert_allclose(q[existing], 0.0, atol=1e-12)
    np.testing.assert_allclose(u[existing], 0.0, atol=1e-12)
    assert np.linalg.norm(q[robot]) > 0.0


def test_prospective_activation_occurs_at_zero_rho_with_outward_rate_guard() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    manager.process(scenario.join_requests[0].time, positions, velocities)

    state = next(iter(manager.prospective.values()))
    state.rho = 0.0
    i, j = state.edge
    positions[i] = positions[j] + np.array([1.0, 0.0])
    cfg = scenario.reconfiguration

    # Fast outward radial motion must delay activation.
    velocities[i] = np.array([cfg.edge_activation_outward_rate_tolerance + 0.2, 0.0])
    velocities[j] = 0.0
    assert not manager._prospective_ready(state, positions, velocities)

    # Closing motion is safe with respect to the upper connectivity boundary.
    velocities[i] = np.array([-0.2, 0.0])
    assert manager._prospective_ready(state, positions, velocities)

    # A positive relaxation, however small, is not activation-ready.
    state.rho = 1e-5
    assert not manager._prospective_ready(state, positions, velocities)



def test_join_preserves_desired_formation_and_uses_bumpless_response_ramp() -> None:
    base = default_open_formation_scenario(dimension=2)
    scenario = replace(
        base,
        reconfiguration=replace(base.reconfiguration, join_handover_duration=2.0),
    )
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    desired_before = manager.current_desired_positions.copy()

    t0 = scenario.join_requests[0].time
    manager.process(t0, positions, velocities)
    state = next(iter(manager.prospective.values()))
    candidate = scenario.join_requests[0].robot
    i, j = state.edge
    # Put the prospective edge safely inside the nominal annulus and remove
    # the relaxation so it activates at the next manager pass.
    existing = j if i == candidate else i
    desired_relative = state.desired_relative
    if i == candidate:
        positions[candidate] = positions[existing] + desired_relative
    else:
        positions[candidate] = positions[existing] - desired_relative
    state.rho = 0.0
    velocities[:] = 0.0
    manager.process(t0 + scenario.dt, positions, velocities)

    assert manager.active[candidate]
    np.testing.assert_array_equal(manager.current_desired_positions, desired_before)
    assert state.edge in manager.join_handovers

    interaction = next(
        item for item in manager.controller_interactions() if item.edge == state.edge
    )
    assert interaction.response_weights is not None
    weights0 = interaction.response_weights
    if i == candidate:
        assert weights0 == pytest.approx((1.0, 0.0), abs=1e-12)
    else:
        assert weights0 == pytest.approx((0.0, 1.0), abs=1e-12)

    # Halfway through the C1 smoothstep, the old-team endpoint has exactly
    # half response while the candidate remains at full response.
    manager.process(t0 + scenario.dt + 1.0, positions, velocities)
    interaction = next(
        item for item in manager.controller_interactions() if item.edge == state.edge
    )
    weights_mid = interaction.response_weights
    assert weights_mid is not None
    if i == candidate:
        assert weights_mid[0] == pytest.approx(1.0)
        assert weights_mid[1] == pytest.approx(0.5, abs=2e-3)
    else:
        assert weights_mid[0] == pytest.approx(0.5, abs=2e-3)
        assert weights_mid[1] == pytest.approx(1.0)

    # Once the ramp expires, the interaction is an ordinary symmetric edge.
    manager.process(t0 + scenario.dt + 2.1, positions, velocities)
    interaction = next(
        item for item in manager.controller_interactions() if item.edge == state.edge
    )
    assert interaction.response_weights is None
    assert state.edge not in manager.join_handovers



def test_join_handover_exposes_exact_pre_and_full_controller_modes() -> None:
    base = default_open_formation_scenario(dimension=2)
    scenario = replace(
        base,
        reconfiguration=replace(base.reconfiguration, join_handover_duration=6.0),
    )
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)

    t0 = scenario.join_requests[0].time
    manager.process(t0, positions, velocities)
    state = next(iter(manager.prospective.values()))
    candidate = scenario.join_requests[0].robot
    i, j = state.edge
    existing = j if i == candidate else i
    if i == candidate:
        positions[candidate] = positions[existing] + state.desired_relative
    else:
        positions[candidate] = positions[existing] - state.desired_relative
    state.rho = 0.0
    manager.process(t0 + scenario.dt, positions, velocities)

    pre = next(
        item for item in manager.controller_interactions("pre") if item.edge == state.edge
    )
    full = next(
        item for item in manager.controller_interactions("full") if item.edge == state.edge
    )
    assert pre.responders == (candidate,)
    assert pre.response_weights is None
    assert full.responders is None
    assert full.response_weights is None

    # The C2 smootherstep is exactly zero at activation, one half at the middle,
    # and one at the end of the handover.
    alpha0, alpha_dot0 = manager.handover_progress(state.edge)
    assert alpha0 == pytest.approx(0.0, abs=1e-12)
    assert alpha_dot0 == pytest.approx(0.0, abs=1e-12)
    manager.process(t0 + scenario.dt + 3.0, positions, velocities)
    alpha_mid, _ = manager.handover_progress(state.edge)
    assert alpha_mid == pytest.approx(0.5, abs=2e-3)


def test_direct_departure_does_not_change_desired_formation() -> None:
    base = default_open_formation_scenario(dimension=2)
    # Construct a leave request for an end robot, whose removal does not need a
    # bridge, but deliberately provide a wildly shifted optional target. The
    # minimal-change policy must ignore it.
    shifted = base.desired_positions + 10.0
    from constrained_omrs.scenario import LeaveRequest
    scenario = replace(
        base,
        join_requests=(),
        leave_requests=(LeaveRequest(robot=0, time=1.0, target_desired_positions=shifted),),
    )
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    desired_before = manager.current_desired_positions.copy()
    manager.process(1.0, positions, velocities)
    manager.process(1.0 + scenario.reconfiguration.min_switch_dwell_time + 0.01, positions, velocities)
    np.testing.assert_array_equal(manager.current_desired_positions, desired_before)


def test_join_guard_prevents_same_cycle_activation_when_rho_starts_at_zero() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    scenario = replace(
        scenario,
        reconfiguration=replace(
            scenario.reconfiguration,
            join_prospective_min_duration=1.0,
            join_activation_error_tolerance=0.50,
        ),
    )
    manager = FormationManager(scenario)
    positions = scenario.initial_positions.copy()
    velocities = np.zeros_like(positions)
    ready = np.ones(scenario.num_robots, dtype=bool)

    t0 = scenario.join_requests[0].time
    manager.process(t0, positions, velocities, join_ready=ready)
    assert manager.prospective
    robot = scenario.join_requests[0].robot
    assert manager.staging[robot]
    assert not manager.active[robot]

    # Put the prospective pair exactly at its desired relative configuration
    # and rho at zero.  Without the explicit prospective dwell, the manager
    # would admit the robot immediately on the next update (or even the same
    # update if assignment already happened in this state).
    positions[:] = scenario.desired_positions
    for state in manager.prospective.values():
        state.rho = 0.0

    manager.process(t0 + 0.25, positions, velocities, join_ready=ready)
    assert manager.staging[robot]
    assert not manager.active[robot]
    assert manager.prospective

    manager.process(t0 + 1.05, positions, velocities, join_ready=ready)
    assert manager.active[robot]
    assert not manager.staging[robot]


def test_prospective_edge_rejects_target_incompatible_with_activation_margin() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    scenario = replace(
        scenario,
        reconfiguration=replace(
            scenario.reconfiguration,
            edge_activation_margin=0.20,
        ),
    )
    manager = FormationManager(scenario)
    desired = scenario.desired_positions.copy()
    # Construct a nominally admissible target whose upper-bound clearance is
    # smaller than the configured protected activation margin.
    edge = (0, scenario.join_requests[0].robot)
    desired[edge[1]] = desired[edge[0]] + np.array(
        [scenario.sensing_distance - 0.10, 0.0]
    )
    positions = scenario.initial_positions.copy()
    positions[edge[1]] = positions[edge[0]] + np.array([1.0, 0.0])

    with pytest.raises(RuntimeError, match="protected activation margin"):
        manager._assign_prospective(edge, desired, positions, 0.0, purpose="join")


def test_departure_bridge_uses_minimum_component_rigid_formation_change() -> None:
    scenario = default_open_formation_scenario(dimension=3)
    manager = FormationManager(scenario)
    positions = scenario.desired_positions.copy()

    # Start the planned removal directly. Robot 2 is an articulation vertex of
    # the initial chain, so a bridge between the two surviving components is
    # required.
    request = scenario.leave_requests[0]
    desired_before = manager.current_desired_positions.copy()
    manager._start_leave(request, positions, request.time)

    assert manager._pending_leave_phase == "forming_bridges"
    assert manager._pending_target_desired is not None
    assert manager.prospective
    target = manager._pending_target_desired

    retained = {
        edge for edge in manager.established_edges if request.robot not in edge
    }
    for i, j in retained:
        before = desired_before[i] - desired_before[j]
        after = target[i] - target[j]
        np.testing.assert_allclose(after, before, atol=1e-12)

    # At least one component must move relative to the other, otherwise the
    # original infeasible bridge geometry would not have changed.
    assert not np.allclose(target, desired_before)

    for state in manager.prospective.values():
        desired_distance = np.linalg.norm(state.desired_relative)
        margin = scenario.reconfiguration.edge_activation_margin
        assert (
            scenario.collision_distance + margin
            < desired_distance
            < scenario.sensing_distance - margin
        )


def test_bridge_activation_is_bumpless_and_defers_node_target_until_leave() -> None:
    scenario = default_open_formation_scenario(dimension=3)
    manager = FormationManager(scenario)
    positions = scenario.desired_positions.copy()
    request = scenario.leave_requests[0]
    desired_before = manager.current_desired_positions.copy()

    manager._start_leave(request, positions, request.time)
    assert manager._pending_target_desired is not None
    target_after_leave = manager._pending_target_desired.copy()
    bridge_states = [
        state for state in manager.prospective.values() if state.purpose == "bridge"
    ]
    assert bridge_states
    bridge_targets = {
        state.edge: state.desired_relative.copy() for state in bridge_states
    }

    manager._activate_bridges(request.time + 1.0)

    # Bridge activation must not alter the node-level desired formation.
    np.testing.assert_array_equal(manager.current_desired_positions, desired_before)
    for edge, desired_relative in bridge_targets.items():
        assert edge in manager.established_edges
        np.testing.assert_allclose(
            manager.desired_relative_for_edge(edge),
            desired_relative,
            atol=1e-12,
        )

    # When robot 2 actually leaves, the component-rigid post-departure target
    # becomes active. It represents exactly the same bridge target, so the
    # override disappears without changing the surviving controller objective.
    manager._finish_leave(request.time + 2.0)
    np.testing.assert_allclose(
        manager.current_desired_positions,
        target_after_leave,
        atol=1e-12,
    )
    assert not manager.established_desired_overrides
    for edge, desired_relative in bridge_targets.items():
        np.testing.assert_allclose(
            manager.desired_relative_for_edge(edge),
            desired_relative,
            atol=1e-12,
        )
