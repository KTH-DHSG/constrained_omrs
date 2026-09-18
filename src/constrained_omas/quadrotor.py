from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


def hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def vee(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    return np.array(
        [matrix[2, 1], matrix[0, 2], matrix[1, 0]],
        dtype=float,
    )


def exp_so3(rotation_vector: np.ndarray) -> np.ndarray:
    phi = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        K = hat(phi)
        return np.eye(3) + K + 0.5 * (K @ K)
    axis = phi / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def project_so3(rotation: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(rotation, dtype=float).reshape(3, 3))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def attitude_error_angle(rotation: np.ndarray, desired_rotation: np.ndarray) -> float:
    relative = desired_rotation.T @ rotation
    cosine = 0.5 * (np.trace(relative) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


@dataclass(frozen=True)
class QuadrotorConfig:
    """Representative medium research-quadrotor model.

    The geometry is used consistently by the rigid-body dynamics, rotor mixer,
    and visual model. ``arm_length`` is the center-to-rotor distance. The
    default vehicle is intentionally generic rather than tied to a proprietary
    platform, but its dimensions and thrust-to-weight ratio are realistic for a
    roughly 1.2 kg laboratory quadrotor.
    """

    mass: float = 1.20
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.018, 0.018, 0.032])
    )
    gravity: float = 9.81
    linear_drag: float = 0.10
    attitude_gain: float = 4.0
    angular_velocity_gain: float = 0.70
    maximum_total_thrust: float = 28.0
    maximum_torque: float = 1.50
    maximum_tilt_deg: float = 35.0
    yaw_reference: float = 0.0

    # Physical geometry used by both the mixer and the animation.
    arm_length: float = 0.23
    body_radius: float = 0.075
    body_height: float = 0.07
    propeller_radius: float = 0.11
    landing_leg_length: float = 0.055

    # Individual rotor / motor model.
    minimum_rotor_thrust: float = 0.0
    maximum_rotor_thrust: float = 7.0
    motor_time_constant: float = 0.045
    yaw_torque_per_thrust: float = 0.018
    rotor_spin_directions: tuple[int, int, int, int] = (1, -1, 1, -1)

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia, dtype=float)
        if inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3x3 matrix.")
        if not np.allclose(inertia, inertia.T, atol=1e-12):
            raise ValueError("inertia must be symmetric.")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("inertia must be positive definite.")
        if self.mass <= 0.0 or self.gravity <= 0.0:
            raise ValueError("mass and gravity must be positive.")
        if self.maximum_total_thrust <= self.mass * self.gravity:
            raise ValueError("maximum_total_thrust must exceed hover thrust.")
        if self.maximum_torque <= 0.0:
            raise ValueError("maximum_torque must be positive.")
        if not 0.0 < self.maximum_tilt_deg < 89.0:
            raise ValueError("maximum_tilt_deg must lie in (0, 89).")
        if self.linear_drag < 0.0:
            raise ValueError("linear_drag must be nonnegative.")
        for name in (
            "arm_length",
            "body_radius",
            "body_height",
            "propeller_radius",
            "landing_leg_length",
            "motor_time_constant",
            "yaw_torque_per_thrust",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.minimum_rotor_thrust < 0.0:
            raise ValueError("minimum_rotor_thrust must be nonnegative.")
        if self.maximum_rotor_thrust <= self.minimum_rotor_thrust:
            raise ValueError("maximum_rotor_thrust must exceed minimum_rotor_thrust.")
        if self.maximum_total_thrust > 4.0 * self.maximum_rotor_thrust + 1e-12:
            raise ValueError(
                "maximum_total_thrust cannot exceed the sum of the four rotor limits."
            )
        if len(self.rotor_spin_directions) != 4 or any(
            direction not in (-1, 1) for direction in self.rotor_spin_directions
        ):
            raise ValueError("rotor_spin_directions must contain four entries in {-1,+1}.")
        if sum(self.rotor_spin_directions) != 0:
            raise ValueError("The default quadrotor requires two CW and two CCW rotors.")
        object.__setattr__(self, "inertia", inertia)

    @property
    def maximum_tilt(self) -> float:
        return float(np.deg2rad(self.maximum_tilt_deg))

    @property
    def hover_rotor_thrust(self) -> float:
        return self.mass * self.gravity / 4.0

    @property
    def rotor_positions_body(self) -> np.ndarray:
        a = self.arm_length / np.sqrt(2.0)
        return np.array(
            [
                [a, a, 0.0],
                [-a, a, 0.0],
                [-a, -a, 0.0],
                [a, -a, 0.0],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class QuadrotorControlOutput:
    thrust: float
    torque: np.ndarray
    desired_rotation: np.ndarray
    desired_force: np.ndarray
    realized_acceleration: np.ndarray
    attitude_error: float


@dataclass(frozen=True)
class RotorAllocationOutput:
    commands: np.ndarray
    raw_commands: np.ndarray
    desired_wrench: np.ndarray
    commanded_wrench: np.ndarray
    saturated: bool


class QuadrotorRotorActuator:
    """Four-rotor thrust allocation with first-order motor dynamics.

    Rotor thrusts are the actuator states. A bounded weighted least-squares
    allocation is solved over the four thrusts. For this small mixer we can
    enumerate the 3^4 possible lower/free/upper active sets, avoiding an
    external QP dependency while still respecting the individual bounds.
    """

    def __init__(self, config: QuadrotorConfig) -> None:
        self.config = config
        positions = config.rotor_positions_body
        spins = np.asarray(config.rotor_spin_directions, dtype=float)

        B = np.zeros((4, 4), dtype=float)
        B[0, :] = 1.0
        B[1, :] = positions[:, 1]          # tau_x = y_i T_i
        B[2, :] = -positions[:, 0]         # tau_y = -x_i T_i
        B[3, :] = spins * config.yaw_torque_per_thrust
        self.allocation_matrix = B

        # Normalize wrench residuals before comparing active-set candidates.
        roll_pitch_scale = max(config.arm_length * config.maximum_total_thrust, 1e-6)
        yaw_scale = max(
            config.yaw_torque_per_thrust * config.maximum_total_thrust,
            1e-6,
        )
        self._weights = np.diag(
            [
                3.0 / config.maximum_total_thrust,
                4.0 / roll_pitch_scale,
                4.0 / roll_pitch_scale,
                0.8 / yaw_scale,
            ]
        )

    def wrench(self, rotor_thrusts: np.ndarray) -> np.ndarray:
        thrusts = np.asarray(rotor_thrusts, dtype=float).reshape(4)
        return self.allocation_matrix @ thrusts

    def _bounded_weighted_least_squares(self, desired_wrench: np.ndarray) -> np.ndarray:
        desired = np.asarray(desired_wrench, dtype=float).reshape(4)
        lower = self.config.minimum_rotor_thrust
        upper = self.config.maximum_rotor_thrust
        B = self.allocation_matrix
        W = self._weights

        best: np.ndarray | None = None
        best_cost = np.inf

        # status: -1 fixed at lower bound, 0 free, +1 fixed at upper bound.
        for status in product((-1, 0, 1), repeat=4):
            status_arr = np.asarray(status, dtype=int)
            free = np.flatnonzero(status_arr == 0)
            fixed = np.flatnonzero(status_arr != 0)
            candidate = np.zeros(4, dtype=float)
            if fixed.size:
                candidate[fixed] = np.where(status_arr[fixed] < 0, lower, upper)

            residual_target = desired - B[:, fixed] @ candidate[fixed] if fixed.size else desired
            if free.size:
                A = W @ B[:, free]
                b = W @ residual_target
                free_solution, *_ = np.linalg.lstsq(A, b, rcond=None)
                if np.any(free_solution < lower - 1e-10) or np.any(
                    free_solution > upper + 1e-10
                ):
                    continue
                candidate[free] = np.clip(free_solution, lower, upper)

            error = W @ (B @ candidate - desired)
            cost = float(error @ error)
            if cost < best_cost:
                best_cost = cost
                best = candidate.copy()

        if best is None:
            # Should never occur because the all-lower and all-upper active sets
            # are always feasible, but keep a deterministic fallback.
            raw = np.linalg.solve(B, desired)
            return np.clip(raw, lower, upper)
        return best

    def allocate(self, collective_thrust: float, torque: np.ndarray) -> RotorAllocationOutput:
        desired = np.concatenate(
            ([float(collective_thrust)], np.asarray(torque, dtype=float).reshape(3))
        )
        raw = np.linalg.solve(self.allocation_matrix, desired)
        saturated = bool(
            np.any(raw < self.config.minimum_rotor_thrust - 1e-10)
            or np.any(raw > self.config.maximum_rotor_thrust + 1e-10)
        )
        commands = (
            self._bounded_weighted_least_squares(desired)
            if saturated
            else raw.copy()
        )
        commanded_wrench = self.wrench(commands)
        return RotorAllocationOutput(
            commands=commands,
            raw_commands=raw,
            desired_wrench=desired,
            commanded_wrench=commanded_wrench,
            saturated=saturated,
        )

    def derivative(
        self,
        rotor_thrusts: np.ndarray,
        rotor_thrust_commands: np.ndarray,
    ) -> np.ndarray:
        thrusts = np.asarray(rotor_thrusts, dtype=float).reshape(4)
        commands = np.asarray(rotor_thrust_commands, dtype=float).reshape(4)
        commands = np.clip(
            commands,
            self.config.minimum_rotor_thrust,
            self.config.maximum_rotor_thrust,
        )
        return (commands - thrusts) / self.config.motor_time_constant

    def integrate(
        self,
        rotor_thrusts: np.ndarray,
        rotor_thrust_commands: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        next_thrusts = np.asarray(rotor_thrusts, dtype=float).reshape(4) + float(dt) * self.derivative(
            rotor_thrusts,
            rotor_thrust_commands,
        )
        return np.clip(
            next_thrusts,
            self.config.minimum_rotor_thrust,
            self.config.maximum_rotor_thrust,
        )


class GeometricQuadrotorController:
    """Map a desired translational acceleration to thrust and body torque.

    The high-level OMRS controller remains unchanged and produces ``a_cmd``.
    This adapter realizes that command through a full rigid-body quadrotor
    model. A fixed desired yaw resolves the attitude about the thrust axis.
    """

    def __init__(self, config: QuadrotorConfig) -> None:
        self.config = config
        self._e3 = np.array([0.0, 0.0, 1.0], dtype=float)

    def _limited_force(self, acceleration_command: np.ndarray) -> np.ndarray:
        cfg = self.config
        a_cmd = np.asarray(acceleration_command, dtype=float).reshape(3)
        force = cfg.mass * (a_cmd + cfg.gravity * self._e3)

        vertical = max(float(force[2]), 0.05 * cfg.mass * cfg.gravity)
        horizontal = force[:2].copy()
        horizontal_norm = float(np.linalg.norm(horizontal))
        maximum_horizontal = vertical * np.tan(cfg.maximum_tilt)
        if horizontal_norm > maximum_horizontal > 0.0:
            horizontal *= maximum_horizontal / horizontal_norm
        force = np.array([horizontal[0], horizontal[1], vertical], dtype=float)

        norm_force = float(np.linalg.norm(force))
        if norm_force > cfg.maximum_total_thrust:
            force *= cfg.maximum_total_thrust / norm_force
        return force

    def desired_rotation(self, acceleration_command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        force = self._limited_force(acceleration_command)
        force_norm = float(np.linalg.norm(force))
        if force_norm <= 1e-12:
            return np.eye(3), force

        b3_des = force / force_norm
        yaw = self.config.yaw_reference
        b1_heading = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        b2_des = np.cross(b3_des, b1_heading)
        b2_norm = float(np.linalg.norm(b2_des))
        if b2_norm <= 1e-9:
            b1_heading = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=float)
            b2_des = np.cross(b3_des, b1_heading)
            b2_norm = float(np.linalg.norm(b2_des))
        b2_des /= b2_norm
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack((b1_des, b2_des, b3_des))
        return project_so3(R_des), force

    def translational_acceleration(
        self,
        rotation: np.ndarray,
        velocity: np.ndarray,
        collective_thrust: float,
    ) -> np.ndarray:
        R = np.asarray(rotation, dtype=float).reshape(3, 3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        body_z = R[:, 2]
        return (
            float(collective_thrust) * body_z / self.config.mass
            - self.config.gravity * self._e3
            - (self.config.linear_drag / self.config.mass) * velocity
        )

    def evaluate(
        self,
        rotation: np.ndarray,
        angular_velocity: np.ndarray,
        velocity: np.ndarray,
        acceleration_command: np.ndarray,
    ) -> QuadrotorControlOutput:
        cfg = self.config
        R = np.asarray(rotation, dtype=float).reshape(3, 3)
        omega = np.asarray(angular_velocity, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)

        R_des, desired_force = self.desired_rotation(acceleration_command)
        e_R = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        gyroscopic = np.cross(omega, cfg.inertia @ omega)
        torque = (
            -cfg.attitude_gain * e_R
            - cfg.angular_velocity_gain * omega
            + gyroscopic
        )
        torque_norm = float(np.linalg.norm(torque))
        if torque_norm > cfg.maximum_torque:
            torque *= cfg.maximum_torque / torque_norm

        body_z = R[:, 2]
        thrust = float(np.clip(desired_force @ body_z, 0.0, cfg.maximum_total_thrust))
        acceleration = self.translational_acceleration(R, velocity, thrust)
        return QuadrotorControlOutput(
            thrust=thrust,
            torque=torque,
            desired_rotation=R_des,
            desired_force=desired_force,
            realized_acceleration=acceleration,
            attitude_error=attitude_error_angle(R, R_des),
        )

    def angular_acceleration(
        self,
        angular_velocity: np.ndarray,
        torque: np.ndarray,
    ) -> np.ndarray:
        omega = np.asarray(angular_velocity, dtype=float).reshape(3)
        torque = np.asarray(torque, dtype=float).reshape(3)
        return np.linalg.solve(
            self.config.inertia,
            torque - np.cross(omega, self.config.inertia @ omega),
        )
