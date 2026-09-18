import numpy as np
import pytest

from constrained_omrs.barrier import EdgeBarrier
from constrained_omrs.controller import (
    ControllerConfig,
    EdgeInteraction,
    SmoothBacksteppingController,
)


@pytest.mark.parametrize(
    "desired_relative,error",
    [
        (np.array([1.2, 0.4]), np.array([0.12, -0.08])),
        (np.array([1.2, 0.4, 0.7]), np.array([0.12, -0.08, 0.05])),
    ],
)
def test_barrier_analytical_hessian_matches_finite_difference(
    desired_relative: np.ndarray,
    error: np.ndarray,
) -> None:
    barrier = EdgeBarrier(
        desired_relative=desired_relative,
        collision_distance=0.5,
        sensing_distance=3.0,
    )
    _, _, hessian = barrier.value_gradient_hessian(error)

    eps = 1e-6
    hessian_fd = np.zeros_like(hessian)
    for j in range(error.size):
        direction = np.zeros_like(error)
        direction[j] = eps
        _, grad_plus = barrier.value_and_gradient(error + direction)
        _, grad_minus = barrier.value_and_gradient(error - direction)
        hessian_fd[:, j] = (grad_plus - grad_minus) / (2.0 * eps)

    np.testing.assert_allclose(hessian, hessian_fd, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("dimension", [2, 3])
def test_virtual_velocity_derivative_includes_relaxation_dynamics(
    dimension: int,
) -> None:
    if dimension == 2:
        desired_positions = np.array(
            [[0.0, 0.0], [1.05, 0.10], [1.85, 0.85]], dtype=float
        )
        perturbation = np.array(
            [[0.05, -0.03], [-0.04, 0.05], [0.03, -0.02]], dtype=float
        )
        velocities = np.array(
            [[0.16, -0.07], [-0.10, 0.05], [0.03, 0.08]], dtype=float
        )
    else:
        desired_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.05, 0.10, 0.25],
                [1.85, 0.85, 0.45],
            ],
            dtype=float,
        )
        perturbation = np.array(
            [
                [0.05, -0.03, 0.02],
                [-0.04, 0.05, -0.01],
                [0.03, -0.02, 0.03],
            ],
            dtype=float,
        )
        velocities = np.array(
            [
                [0.16, -0.07, 0.02],
                [-0.10, 0.05, -0.01],
                [0.03, 0.08, 0.04],
            ],
            dtype=float,
        )

    positions = desired_positions + perturbation
    active = np.ones(3, dtype=bool)
    rho = 0.35
    rho_dot = -0.12
    interactions = [
        EdgeInteraction(
            edge=(0, 1),
            desired_relative=desired_positions[0] - desired_positions[1],
        ),
        EdgeInteraction(
            edge=(1, 2),
            desired_relative=desired_positions[1] - desired_positions[2],
            relaxation=rho,
            relaxation_rate=rho_dot,
            prospective=True,
        ),
    ]

    controller = SmoothBacksteppingController(
        num_robots=3,
        dimension=dimension,
        collision_distance=0.45,
        sensing_distance=1.60,
        config=ControllerConfig(k1=1.1, k2=0.8, k3=2.0, max_virtual_speed=0.8),
    )

    analytical = controller.virtual_velocity_derivative(
        positions,
        velocities,
        interactions,
        active,
    )

    eps = 1e-6

    def shifted_interactions(sign: float) -> list[EdgeInteraction]:
        return [
            interactions[0],
            EdgeInteraction(
                edge=(1, 2),
                desired_relative=interactions[1].desired_relative,
                relaxation=rho + sign * eps * rho_dot,
                relaxation_rate=rho_dot,
                prospective=True,
            ),
        ]

    v_plus, _, _ = controller.virtual_velocity(
        positions + eps * velocities,
        shifted_interactions(+1.0),
        active,
    )
    v_minus, _, _ = controller.virtual_velocity(
        positions - eps * velocities,
        shifted_interactions(-1.0),
        active,
    )
    numerical = (v_plus - v_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytical, numerical, rtol=4e-5, atol=5e-7)


def test_virtual_velocity_derivative_includes_collision_only_pair() -> None:
    desired = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.1], [0.55, 0.70, 0.05]],
        dtype=float,
    )
    positions = desired + np.array(
        [[0.02, -0.01, 0.01], [-0.03, 0.01, 0.0], [0.01, -0.02, 0.01]]
    )
    velocities = np.array(
        [[0.05, -0.03, 0.01], [-0.04, 0.02, -0.01], [0.02, 0.01, 0.0]]
    )
    controlled = np.ones(3, dtype=bool)
    interactions = [
        EdgeInteraction(edge=(0, 1), desired_relative=desired[0] - desired[1])
    ]
    collision_pairs = [(0, 2)]
    controller = SmoothBacksteppingController(
        num_robots=3,
        dimension=3,
        collision_distance=0.45,
        sensing_distance=1.65,
        config=ControllerConfig(
            k1=0.7, k2=0.4, k3=1.2, max_virtual_speed=0.5,
            collision_weight=0.02,
        ),
    )

    analytical = controller.virtual_velocity_derivative(
        positions, velocities, interactions, controlled, collision_pairs
    )
    eps = 1e-6
    v_plus, _, _ = controller.virtual_velocity(
        positions + eps * velocities, interactions, controlled, collision_pairs
    )
    v_minus, _, _ = controller.virtual_velocity(
        positions - eps * velocities, interactions, controlled, collision_pairs
    )
    numerical = (v_plus - v_minus) / (2.0 * eps)
    np.testing.assert_allclose(analytical, numerical, rtol=5e-5, atol=8e-7)


def test_nearby_collision_pairs_exclude_graph_edges() -> None:
    controller = SmoothBacksteppingController(
        num_robots=4,
        dimension=3,
        collision_distance=0.45,
        sensing_distance=1.65,
        config=ControllerConfig(),
    )
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.8, 0.0], [4.0, 0.0, 0.0]]
    )
    participants = np.array([True, True, True, False])
    pairs = controller.nearby_collision_pairs(
        positions, participants, excluded_edges={(0, 1)}
    )
    assert set(pairs) == {(0, 2), (1, 2)}


def test_virtual_velocity_derivative_includes_smooth_response_handover() -> None:
    desired = np.array(
        [[0.0, 0.0, 0.0], [1.10, 0.20, 0.15]], dtype=float
    )
    positions = desired + np.array(
        [[0.03, -0.02, 0.01], [-0.04, 0.03, -0.02]], dtype=float
    )
    velocities = np.array(
        [[0.08, -0.03, 0.02], [-0.05, 0.04, -0.01]], dtype=float
    )
    controlled = np.ones(2, dtype=bool)
    alpha = 0.37
    alpha_dot = 0.42
    interaction = EdgeInteraction(
        edge=(0, 1),
        desired_relative=desired[0] - desired[1],
        response_weights=(1.0, alpha),
        response_weight_rates=(0.0, alpha_dot),
    )
    controller = SmoothBacksteppingController(
        num_robots=2,
        dimension=3,
        collision_distance=0.45,
        sensing_distance=1.65,
        config=ControllerConfig(k1=0.6, k2=0.4, k3=1.2, max_virtual_speed=0.5),
    )

    analytical = controller.virtual_velocity_derivative(
        positions, velocities, [interaction], controlled
    )
    eps = 1e-6

    def interaction_at(sign: float) -> EdgeInteraction:
        return EdgeInteraction(
            edge=(0, 1),
            desired_relative=desired[0] - desired[1],
            response_weights=(1.0, alpha + sign * eps * alpha_dot),
            response_weight_rates=(0.0, alpha_dot),
        )

    v_plus, _, _ = controller.virtual_velocity(
        positions + eps * velocities, [interaction_at(+1.0)], controlled
    )
    v_minus, _, _ = controller.virtual_velocity(
        positions - eps * velocities, [interaction_at(-1.0)], controlled
    )
    numerical = (v_plus - v_minus) / (2.0 * eps)
    np.testing.assert_allclose(analytical, numerical, rtol=5e-5, atol=8e-7)
