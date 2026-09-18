from __future__ import annotations

import numpy as np


def quaternion_to_rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the world-from-body rotation matrix for an ``[x,y,z,w]`` quaternion."""
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be positive.")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def odometry_body_twist_to_world_linear_velocity(
    rotation_world_from_body: np.ndarray,
    linear_velocity_body: np.ndarray,
) -> np.ndarray:
    """Convert Gazebo OdometryPublisher linear twist from body to world coordinates.

    Gazebo Harmonic's 3-D ``OdometryPublisher`` expresses the twist in the
    robot-base frame. The OMRS controller, in contrast, uses inertial/world
    translational velocities.  Angular velocity is already in the body frame,
    which is exactly what the geometric attitude controller requires.
    """
    rotation = np.asarray(rotation_world_from_body, dtype=float).reshape(3, 3)
    velocity = np.asarray(linear_velocity_body, dtype=float).reshape(3)
    return rotation @ velocity


def rotor_thrusts_to_motor_speeds(
    rotor_thrusts: np.ndarray,
    motor_constant: float,
    maximum_rotor_velocity: float,
) -> np.ndarray:
    """Map rotor thrust commands [N] to Gazebo motor-speed commands [rad/s].

    ``MulticopterMotorModel`` uses the standard quadratic thrust model
    ``T = k_f * omega^2``.  This function clips numerical undershoot below zero
    and respects the motor-model angular-speed limit.
    """
    thrusts = np.asarray(rotor_thrusts, dtype=float).reshape(-1)
    if motor_constant <= 0.0:
        raise ValueError("motor_constant must be positive.")
    if maximum_rotor_velocity <= 0.0:
        raise ValueError("maximum_rotor_velocity must be positive.")
    speeds = np.sqrt(np.maximum(thrusts, 0.0) / float(motor_constant))
    return np.clip(speeds, 0.0, float(maximum_rotor_velocity))


def motor_speed_for_hover(
    mass: float,
    gravity: float,
    motor_constant: float,
    num_rotors: int = 4,
) -> float:
    if mass <= 0.0 or gravity <= 0.0:
        raise ValueError("mass and gravity must be positive.")
    if num_rotors <= 0:
        raise ValueError("num_rotors must be positive.")
    return float(np.sqrt((mass * gravity / num_rotors) / motor_constant))
