import numpy as np
import pytest

from constrained_omrs.gazebo_adapter import (
    motor_speed_for_hover,
    odometry_body_twist_to_world_linear_velocity,
    quaternion_to_rotation_xyzw,
    rotor_thrusts_to_motor_speeds,
)


def test_quaternion_rotation_and_body_velocity_conversion() -> None:
    # +90 deg yaw: body x points along world y.
    q = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    rotation = quaternion_to_rotation_xyzw(q)
    world_velocity = odometry_body_twist_to_world_linear_velocity(
        rotation,
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(world_velocity, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_rotor_thrust_to_motor_speed_matches_quadratic_model() -> None:
    kf = 8.641975308641975e-6
    thrusts = np.array([0.0, 1.0, 3.0, 7.0])
    speeds = rotor_thrusts_to_motor_speeds(thrusts, kf, 900.0)
    np.testing.assert_allclose(kf * speeds**2, thrusts, rtol=1e-12, atol=1e-12)


def test_motor_speed_is_clipped_to_plugin_limit() -> None:
    speeds = rotor_thrusts_to_motor_speeds(
        np.array([100.0]),
        motor_constant=8.0e-6,
        maximum_rotor_velocity=900.0,
    )
    assert speeds[0] == pytest.approx(900.0)


def test_hover_speed_reconstructs_vehicle_weight() -> None:
    mass = 1.2
    gravity = 9.81
    kf = 7.0 / 900.0**2
    omega = motor_speed_for_hover(mass, gravity, kf)
    assert 4.0 * kf * omega**2 == pytest.approx(mass * gravity)
