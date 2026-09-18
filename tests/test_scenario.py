import numpy as np
import pytest

from constrained_omrs import default_open_formation_scenario
from constrained_omrs.graph import is_connected


@pytest.mark.parametrize("dimension", [2, 3])
def test_default_scenario_has_requested_dimension(dimension: int) -> None:
    scenario = default_open_formation_scenario(dimension=dimension)
    assert scenario.dimension == dimension
    assert scenario.initial_positions.shape == scenario.desired_positions.shape
    assert scenario.initial_velocities.shape == scenario.desired_positions.shape

    if dimension == 2:
        assert scenario.num_robots == 5
        assert np.count_nonzero(scenario.initial_active) == 4
        assert len(scenario.join_requests) == 1
        assert len(scenario.leave_requests) == 1
    else:
        assert scenario.num_robots == 7
        assert np.count_nonzero(scenario.initial_active) == 5
        assert len(scenario.join_requests) == 2
        assert len(scenario.leave_requests) == 3
        assert scenario.charging_area is not None
        np.testing.assert_allclose(
            scenario.initial_positions[5:],
            scenario.charging_area.pad_positions[5:],
        )


def test_default_scenario_rejects_other_dimensions() -> None:
    with pytest.raises(ValueError):
        default_open_formation_scenario(dimension=4)


@pytest.mark.parametrize("dimension", [2, 3])
def test_initial_graph_is_connected_and_admissible(dimension: int) -> None:
    scenario = default_open_formation_scenario(dimension=dimension)
    vertices = np.flatnonzero(scenario.initial_active)
    assert is_connected(vertices, scenario.initial_edges)
    for i, j in scenario.initial_edges:
        distance = np.linalg.norm(scenario.initial_positions[i] - scenario.initial_positions[j])
        assert scenario.collision_distance < distance < scenario.sensing_distance


def test_default_2d_requests_demonstrate_join_and_make_before_break_removal() -> None:
    scenario = default_open_formation_scenario(dimension=2)
    join = scenario.join_requests[0]
    leave = scenario.leave_requests[0]
    assert join.robot == 4
    assert join.time == pytest.approx(6.0)
    assert leave.robot == 1
    assert leave.time == pytest.approx(18.0)
    assert leave.target_desired_positions is not None

    # Removing robot 1 disconnects the initial chain. The old desired 0--2
    # distance is deliberately outside d_max, while the new formation makes the
    # required bridge feasible.
    old_bridge_distance = np.linalg.norm(
        scenario.desired_positions[0] - scenario.desired_positions[2]
    )
    new_bridge_distance = np.linalg.norm(
        leave.target_desired_positions[0] - leave.target_desired_positions[2]
    )
    assert old_bridge_distance > scenario.sensing_distance
    assert scenario.collision_distance < new_bridge_distance < scenario.sensing_distance


def test_default_3d_scenario_is_spatial_and_requires_a_bridge_before_removal() -> None:
    scenario = default_open_formation_scenario(dimension=3)
    assert np.ptp(scenario.desired_positions[:, 2]) > 1.0

    first_join, second_join = scenario.join_requests
    assert (first_join.robot, first_join.num_edges) == (5, 2)
    assert (second_join.robot, second_join.num_edges) == (6, 2)
    assert first_join.time == pytest.approx(8.0)
    assert second_join.time == pytest.approx(25.0)

    # The initial active team is intentionally displaced from the desired
    # relative formation so the fixed-team formation controller has a visible
    # convergence transient before any membership change.
    edge_errors = []
    for i, j in scenario.initial_edges:
        actual = scenario.initial_positions[i] - scenario.initial_positions[j]
        desired = scenario.desired_positions[i] - scenario.desired_positions[j]
        edge_errors.append(np.linalg.norm(actual - desired))
    assert min(edge_errors) > 0.55

    assert [request.robot for request in scenario.leave_requests] == [2, 1, 3]
    assert [request.time for request in scenario.leave_requests] == pytest.approx(
        [51.0, 73.0, 95.0]
    )
    leave = scenario.leave_requests[0]
    assert leave.target_desired_positions is not None

    old_bridge_distance = np.linalg.norm(
        scenario.desired_positions[1] - scenario.desired_positions[3]
    )
    new_bridge_distance = np.linalg.norm(
        leave.target_desired_positions[1] - leave.target_desired_positions[3]
    )
    assert old_bridge_distance > scenario.sensing_distance
    assert scenario.collision_distance < new_bridge_distance < scenario.sensing_distance
