import numpy as np
import pytest

from constrained_omrs.barrier import EdgeBarrier


@pytest.mark.parametrize(
    "desired_relative",
    [
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0, 0.5]),
    ],
)
def test_barrier_zero_at_desired_edge(desired_relative: np.ndarray) -> None:
    barrier = EdgeBarrier(
        desired_relative=desired_relative,
        collision_distance=0.5,
        sensing_distance=3.0,
    )
    value, gradient = barrier.value_and_gradient(np.zeros_like(desired_relative))
    assert abs(value) < 1e-12
    assert np.linalg.norm(gradient) < 1e-12


@pytest.mark.parametrize(
    "desired_relative,error",
    [
        (np.array([1.15, 0.25]), np.array([0.18, -0.06])),
        (np.array([1.15, 0.25, 0.30]), np.array([0.18, -0.06, 0.04])),
    ],
)
def test_gradient_upper_radius_derivative_matches_finite_difference(
    desired_relative: np.ndarray,
    error: np.ndarray,
) -> None:
    collision_distance = 0.45
    upper_radius = 2.2
    barrier = EdgeBarrier(
        desired_relative=desired_relative,
        collision_distance=collision_distance,
        sensing_distance=upper_radius,
    )
    analytical = barrier.gradient_upper_radius_derivative(error)

    eps = 1e-6
    plus = EdgeBarrier(
        desired_relative=desired_relative,
        collision_distance=collision_distance,
        sensing_distance=upper_radius + eps,
    )
    minus = EdgeBarrier(
        desired_relative=desired_relative,
        collision_distance=collision_distance,
        sensing_distance=upper_radius - eps,
    )
    _, grad_plus = plus.value_and_gradient(error)
    _, grad_minus = minus.value_and_gradient(error)
    numerical = (grad_plus - grad_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-7)


def test_collision_only_barrier_is_compact_and_repulsive() -> None:
    from constrained_omrs.barrier import CollisionBarrier

    barrier = CollisionBarrier(
        collision_distance=0.45,
        activation_distance=1.65,
        weight=0.02,
    )

    # Inside the sensing radius, the gradient points toward decreasing
    # distance; therefore the controller term -grad is repulsive.
    delta = np.array([0.80, 0.0])
    value, gradient, hessian = barrier.value_gradient_hessian(delta)
    assert value > 0.0
    assert gradient[0] < 0.0
    assert hessian.shape == (2, 2)

    # At and outside d_max the collision-only potential is exactly inactive.
    value, gradient, hessian = barrier.value_gradient_hessian(np.array([1.65, 0.0]))
    assert value == 0.0
    np.testing.assert_allclose(gradient, 0.0)
    np.testing.assert_allclose(hessian, 0.0)


def test_collision_only_barrier_hessian_matches_finite_difference() -> None:
    from constrained_omrs.barrier import CollisionBarrier

    barrier = CollisionBarrier(0.45, 1.65, weight=0.02)
    delta = np.array([0.72, -0.31, 0.12])
    _, _, hessian = barrier.value_gradient_hessian(delta)

    eps = 1e-6
    numerical = np.zeros_like(hessian)
    for j in range(delta.size):
        direction = np.zeros_like(delta)
        direction[j] = eps
        _, g_plus = barrier.value_and_gradient(delta + direction)
        _, g_minus = barrier.value_and_gradient(delta - direction)
        numerical[:, j] = (g_plus - g_minus) / (2.0 * eps)

    np.testing.assert_allclose(hessian, numerical, rtol=3e-5, atol=3e-7)
