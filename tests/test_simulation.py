from dataclasses import replace

import numpy as np
import pytest

from constrained_omrs import (
    ControllerConfig,
    SmoothBacksteppingController,
    default_open_formation_scenario,
    simulate,
)
from constrained_omrs.graph import is_connected


def make_controller(scenario, speed_limit: float = 0.8) -> SmoothBacksteppingController:
    return SmoothBacksteppingController(
        num_robots=scenario.num_robots,
        dimension=scenario.dimension,
        collision_distance=scenario.collision_distance,
        sensing_distance=scenario.sensing_distance,
        config=ControllerConfig(
            k1=1.0,
            k2=0.8,
            k3=2.5,
            max_virtual_speed=speed_limit,
        ),
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_short_simulation_runs_and_keeps_virtual_speed_bounded(dimension: int) -> None:
    scenario = replace(default_open_formation_scenario(dimension=dimension), duration=0.2)
    controller = make_controller(scenario)
    result = simulate(scenario, controller)

    assert result.positions.shape[2] == dimension
    assert result.velocities.shape[2] == dimension
    assert result.controls.shape[2] == dimension
    virtual_speed = np.linalg.norm(result.virtual_velocities, axis=2)
    assert np.nanmax(virtual_speed) <= controller.config.max_virtual_speed + 1e-12


def _assert_safe_and_connected(scenario, result) -> None:
    for active, edges in zip(result.active, result.edge_history):
        assert is_connected(np.flatnonzero(active), edges)

    assert np.nanmin(result.min_pair_distance) > scenario.collision_distance
    assert np.nanmax(result.max_established_edge_distance) < scenario.sensing_distance

    margins = [
        margin
        for history in result.prospective_margin_history
        for margin in history.values()
    ]
    assert margins
    assert min(margins) > -2e-8


def test_full_2d_reconfiguration_sequence_is_safe_and_connected() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    result = simulate(scenario, make_controller(scenario))

    kinds = [event.kind for event in result.switch_events]
    assert kinds == ["robot_join", "bridge_activation", "robot_leave"]

    join_event, bridge_event, leave_event = result.switch_events
    assert join_event.edges_added == ((3, 4),)
    assert bridge_event.edges_added == ((0, 2),)
    assert set(leave_event.edges_removed) == {(0, 1), (1, 2)}
    assert leave_event.time - bridge_event.time >= (
        scenario.reconfiguration.min_switch_dwell_time - scenario.dt
    )

    _assert_safe_and_connected(scenario, result)


def test_full_3d_reconfiguration_sequence_is_safe_and_connected() -> None:
    scenario = default_open_formation_scenario(dimension=3)
    result = simulate(scenario, make_controller(scenario))

    kinds = [event.kind for event in result.switch_events]
    assert kinds == [
        "robot_join",
        "robot_join",
        "bridge_activation",
        "robot_leave",
        "bridge_activation",
        "robot_leave",
        "bridge_activation",
        "robot_leave",
    ]

    first_join, second_join = result.switch_events[:2]
    assert set(first_join.edges_added) == {(3, 5), (4, 5)}
    assert set(second_join.edges_added) == {(0, 6), (1, 6)}

    bridge_events = [event for event in result.switch_events if event.kind == "bridge_activation"]
    leave_events = [event for event in result.switch_events if event.kind == "robot_leave"]
    assert [event.robot for event in leave_events] == [2, 1, 3]
    assert [event.edges_added for event in bridge_events] == [
        ((1, 3),),
        ((0, 3),),
        ((0, 4),),
    ]
    for bridge_event, leave_event in zip(bridge_events, leave_events):
        assert leave_event.time - bridge_event.time >= (
            scenario.reconfiguration.min_switch_dwell_time - scenario.dt
        )

    # Every make-before-break bridge must spend a genuine prospective interval
    # in the controller before activation.
    for bridge_event in bridge_events:
        edge = bridge_event.edges_added[0]
        prospective_bridge_samples = sum(
            edge in history for history in result.prospective_history
        )
        assert prospective_bridge_samples > 10

    _assert_safe_and_connected(scenario, result)

    # Every departed articulation robot completes its independent return-to-
    # charge service flight.
    assert scenario.charging_area is not None
    for robot in (1, 2, 3):
        assert result.parked[-1, robot]
        assert not result.returning[-1, robot]
        np.testing.assert_allclose(
            result.positions[-1, robot],
            scenario.charging_area.pad_positions[robot],
            atol=1e-12,
        )


def test_node_formation_update_is_deferred_until_robot_departure() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    result = simulate(scenario, make_controller(scenario))
    bridge_event = next(
        event for event in result.switch_events if event.kind == "bridge_activation"
    )
    leave_event = next(
        event for event in result.switch_events if event.kind == "robot_leave"
    )

    before_bridge = np.flatnonzero(
        result.time < bridge_event.time - 0.5 * scenario.dt
    )[-1]
    after_bridge = np.flatnonzero(
        result.time >= bridge_event.time - 0.5 * scenario.dt
    )[0]
    after_leave = np.flatnonzero(
        result.time >= leave_event.time - 0.5 * scenario.dt
    )[0]

    # Bridge activation itself must not rewrite the node-level formation.
    np.testing.assert_allclose(
        result.desired_positions[before_bridge], scenario.desired_positions
    )
    np.testing.assert_allclose(
        result.desired_positions[after_bridge], scenario.desired_positions
    )

    # The post-departure formation is allowed to change, but only through
    # component-rigid translations. Every surviving established edge therefore
    # keeps exactly the same desired relative displacement.
    assert not np.allclose(
        result.desired_positions[after_leave], scenario.desired_positions
    )
    bridge_edges = {
        edge
        for event in result.switch_events
        if event.kind == "bridge_activation"
        for edge in event.edges_added
    }
    surviving_edges = [
        edge
        for edge in result.edge_history[after_leave]
        if leave_event.robot not in edge and edge not in bridge_edges
    ]
    for i, j in surviving_edges:
        before = scenario.desired_positions[i] - scenario.desired_positions[j]
        after = (
            result.desired_positions[after_leave, i]
            - result.desired_positions[after_leave, j]
        )
        np.testing.assert_allclose(after, before, atol=1e-10)
