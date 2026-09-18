from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChargingAreaConfig:
    """Charging-pad geometry and service-flight policies.

    ``pad_positions`` stores the vehicle-center pose when a drone is parked on
    its charging pad.  Besides the return-to-charge controller, the same smooth
    bounded point-to-point controller is used by the Gazebo integration for
    takeoff / deployment phases that happen outside the OMRS dynamics.
    """

    pad_positions: np.ndarray
    approach_height: float = 0.85
    approach_radius: float = 0.45
    maximum_speed: float = 1.25
    maximum_descent_speed: float = 0.55
    position_gain: float = 0.55
    velocity_gain: float = 2.2
    maximum_acceleration: float = 3.5
    landing_position_tolerance: float = 0.09
    landing_speed_tolerance: float = 0.18
    platform_padding: float = 0.32

    def __post_init__(self) -> None:
        pads = np.asarray(self.pad_positions, dtype=float)
        if pads.ndim != 2 or pads.shape[1] != 3:
            raise ValueError("pad_positions must have shape (num_robots, 3).")
        if not np.all(np.isfinite(pads)):
            raise ValueError("pad_positions must be finite.")
        for name in (
            "approach_height",
            "approach_radius",
            "maximum_speed",
            "maximum_descent_speed",
            "position_gain",
            "velocity_gain",
            "maximum_acceleration",
            "landing_position_tolerance",
            "landing_speed_tolerance",
            "platform_padding",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        object.__setattr__(self, "pad_positions", pads)

    def pad(self, robot: int) -> np.ndarray:
        return self.pad_positions[int(robot)].copy()

    def takeoff_waypoint(self, robot: int, height: float | None = None) -> np.ndarray:
        """Return a vertical-clearance waypoint above a charging pad."""
        target = self.pad(robot)
        target[2] += self.approach_height if height is None else float(height)
        return target

    def point_acceleration_command(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target: np.ndarray,
        *,
        maximum_speed: float | None = None,
        maximum_acceleration: float | None = None,
        maximum_descent_speed: float | None = None,
        position_gain: float | None = None,
        velocity_gain: float | None = None,
    ) -> np.ndarray:
        """Smooth bounded acceleration command toward an arbitrary 3-D target.

        This intentionally remains a small service controller, separate from
        the paper's OMRS controller.  It is used only before admission to the
        interaction graph, during initial deployment, and after departure.
        """
        position = np.asarray(position, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)

        speed_limit = self.maximum_speed if maximum_speed is None else float(maximum_speed)
        acceleration_limit = (
            self.maximum_acceleration
            if maximum_acceleration is None
            else float(maximum_acceleration)
        )
        if speed_limit <= 0.0 or acceleration_limit <= 0.0:
            raise ValueError("Service-flight limits must be positive.")

        kp = self.position_gain if position_gain is None else float(position_gain)
        kv = self.velocity_gain if velocity_gain is None else float(velocity_gain)
        if kp <= 0.0 or kv <= 0.0:
            raise ValueError("Service-flight gains must be positive.")

        # A saturated proportional velocity reference followed by a velocity
        # feedback loop gives a deliberately overdamped point-to-point
        # controller near the target.  This is less aggressive than the old
        # tanh field and is much better suited to the actuator / attitude lag
        # of the Gazebo quadrotor.
        error = target - position
        desired_velocity = kp * error
        desired_speed = float(np.linalg.norm(desired_velocity))
        if desired_speed > speed_limit:
            desired_velocity *= speed_limit / desired_speed

        if maximum_descent_speed is not None:
            descent_limit = float(maximum_descent_speed)
            if descent_limit <= 0.0:
                raise ValueError("maximum_descent_speed must be positive.")
            desired_velocity[2] = max(desired_velocity[2], -descent_limit)

        acceleration = kv * (desired_velocity - velocity)
        norm = float(np.linalg.norm(acceleration))
        if norm > acceleration_limit:
            acceleration *= acceleration_limit / norm
        return acceleration

    def target_reached(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target: np.ndarray,
        *,
        position_tolerance: float,
        speed_tolerance: float,
    ) -> bool:
        position = np.asarray(position, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)
        return (
            np.linalg.norm(position - target) <= float(position_tolerance)
            and np.linalg.norm(velocity) <= float(speed_tolerance)
        )

    def acceleration_command(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        robot: int,
    ) -> np.ndarray:
        """Return a bounded acceleration command for a returning drone.

        The drone first translates toward a point above its pad. Once the
        horizontal error is small, the reference descends to the pad.
        """
        position = np.asarray(position, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        pad = self.pad(robot)

        horizontal_error = float(np.linalg.norm(position[:2] - pad[:2]))
        target = pad.copy()
        if horizontal_error > self.approach_radius:
            target[2] = max(pad[2] + self.approach_height, position[2] * 0.65)

        return self.point_acceleration_command(
            position,
            velocity,
            target,
            maximum_descent_speed=self.maximum_descent_speed,
        )

    def landed(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        robot: int,
    ) -> bool:
        return self.target_reached(
            position,
            velocity,
            self.pad(robot),
            position_tolerance=self.landing_position_tolerance,
            speed_tolerance=self.landing_speed_tolerance,
        )
