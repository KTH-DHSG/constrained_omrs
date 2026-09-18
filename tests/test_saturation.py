import numpy as np
import pytest

from constrained_omrs.controller import SmoothBacksteppingController


@pytest.mark.parametrize("dimension", [2, 3])
def test_radial_arctan_saturation_respects_norm_limit(dimension: int) -> None:
    z = np.zeros((3, dimension))
    z[0] = np.arange(1, dimension + 1, dtype=float) * 4.0
    z[1] = np.arange(1, dimension + 1, dtype=float) * 0.1
    limit = 0.8
    saturated = SmoothBacksteppingController.radial_arctan_saturation(z, limit)
    assert np.all(np.linalg.norm(saturated, axis=1) <= limit + 1e-12)


@pytest.mark.parametrize("dimension", [2, 3])
def test_radial_arctan_saturation_is_locally_identity(dimension: int) -> None:
    z = np.arange(1, dimension + 1, dtype=float)[None, :] * 1e-7
    z[:, 1::2] *= -1.0
    saturated = SmoothBacksteppingController.radial_arctan_saturation(z, 0.8)
    assert np.allclose(saturated, z, rtol=1e-10, atol=1e-15)


def test_smooth_damping_terms_are_bounded_and_dissipative() -> None:
    from constrained_omrs.controller import ControllerConfig, EdgeInteraction

    controller = SmoothBacksteppingController(
        num_robots=3,
        dimension=3,
        collision_distance=0.45,
        sensing_distance=1.65,
        config=ControllerConfig(
            k1=0.3,
            k2=0.7,
            k3=1.5,
            max_virtual_speed=0.35,
            max_edge_damping=0.22,
            max_local_damping=0.38,
        ),
    )
    desired = np.array(
        [[-0.8, 0.0, 1.0], [0.0, 0.4, 1.2], [0.8, 0.0, 1.1]],
        dtype=float,
    )
    positions = desired.copy()
    velocities = np.array(
        [[1.1, -0.4, 0.3], [-0.6, 0.7, -0.2], [0.2, -0.8, 0.5]],
        dtype=float,
    )
    interactions = [
        EdgeInteraction((0, 1), desired[0] - desired[1]),
        EdgeInteraction((1, 2), desired[1] - desired[2]),
    ]
    out = controller.control(
        positions, velocities, interactions, np.ones(3, dtype=bool)
    )

    edge_velocity_error = np.asarray(out["edge_velocity_error"])
    edge_damping = np.asarray(out["edge_damping"])
    local_damping = np.asarray(out["local_damping"])
    v_tilde = np.asarray(out["v_tilde"])

    assert np.max(np.linalg.norm(edge_damping, axis=1)) <= 0.22 + 1e-12
    assert np.max(np.linalg.norm(local_damping, axis=1)) <= 0.38 + 1e-12
    assert float(np.sum(edge_velocity_error * edge_damping)) >= -1e-12
    assert float(np.sum(v_tilde * local_damping)) >= -1e-12
