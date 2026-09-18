from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeBarrier:
    """Weight-recentered BLF for one collision/connectivity edge.

    The physical relative displacement is ``delta = e + b``, where ``b`` is
    the desired relative displacement.  The admissible annulus is

        collision_distance < ||delta|| < sensing_distance.

    For a prospective edge, ``sensing_distance`` is simply replaced by the
    relaxed upper radius ``d_max + rho``.  Consequently the exact same barrier
    implementation is used for established and prospective interactions.
    """

    desired_relative: np.ndarray
    collision_distance: float
    sensing_distance: float

    def __post_init__(self) -> None:
        b = np.asarray(self.desired_relative, dtype=float).reshape(-1)
        object.__setattr__(self, "desired_relative", b)
        if self.collision_distance <= 0.0:
            raise ValueError("collision_distance must be positive.")
        if self.sensing_distance <= self.collision_distance:
            raise ValueError("sensing_distance must exceed collision_distance.")

        b_norm = float(np.linalg.norm(b))
        if not self.collision_distance < b_norm < self.sensing_distance:
            raise ValueError(
                "The desired edge length must lie strictly inside the barrier annulus: "
                f"{self.collision_distance} < {b_norm:.6g} < {self.sensing_distance}."
            )

    @property
    def _b2(self) -> float:
        return float(self.desired_relative @ self.desired_relative)

    @property
    def _kappa_upper(self) -> float:
        d2 = self.collision_distance**2
        b2 = self._b2
        return 0.5 * d2 / (b2 * (b2 - d2))

    @property
    def _kappa_lower(self) -> float:
        r2 = self.sensing_distance**2
        return 0.5 / (r2 - self._b2)

    def _check_domain(self, delta: np.ndarray) -> float:
        s = float(delta @ delta)
        d2 = self.collision_distance**2
        r2 = self.sensing_distance**2
        if not d2 < s < r2:
            raise FloatingPointError(
                "Barrier domain violated: expected collision_distance < ||delta|| "
                f"< sensing_distance, got ||delta||={np.sqrt(max(s, 0.0)):.6g}, "
                f"upper radius={self.sensing_distance:.6g}."
            )
        return s

    def value_and_gradient(self, error: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = self.value_gradient_hessian(error)
        return value, gradient

    def value_gradient_hessian(
        self,
        error: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return the BLF value, gradient and Hessian with respect to ``e``."""
        error = np.asarray(error, dtype=float).reshape(-1)
        delta = error + self.desired_relative
        s = self._check_domain(delta)

        d2 = self.collision_distance**2
        r2 = self.sensing_distance**2
        b2 = self._b2
        ku = self._kappa_upper
        kl = self._kappa_lower

        upper = np.log(r2 / (r2 - s)) - np.log(r2 / (r2 - b2))
        lower = np.log(s / (s - d2)) - np.log(b2 / (b2 - d2))
        value = 0.5 * float(error @ error) + ku * upper + kl * lower

        c = 2.0 * ku / (r2 - s) - 2.0 * kl * d2 / (s * (s - d2))
        gradient = error + c * delta

        c_prime = (
            2.0 * ku / (r2 - s) ** 2
            + 2.0
            * kl
            * d2
            * (2.0 * s - d2)
            / (s**2 * (s - d2) ** 2)
        )
        hessian = (1.0 + c) * np.eye(error.size) + 2.0 * c_prime * np.outer(
            delta, delta
        )
        return float(value), gradient, hessian

    def gradient_upper_radius_derivative(self, error: np.ndarray) -> np.ndarray:
        """Return ``partial(grad_e W)/partial R`` for ``R=sensing_distance``.

        This term is needed for the exact continuous-time derivative of the
        virtual velocity when a prospective edge has a time-varying relaxed
        radius ``R=d_max+rho``.
        """
        error = np.asarray(error, dtype=float).reshape(-1)
        delta = error + self.desired_relative
        s = self._check_domain(delta)

        d2 = self.collision_distance**2
        b2 = self._b2
        R = self.sensing_distance
        r2 = R**2
        ku = self._kappa_upper

        # grad W = e + c(s,R) delta. Only c depends explicitly on R.
        c_R = (
            -4.0 * ku * R / (r2 - s) ** 2
            + 2.0 * R * d2 / ((r2 - b2) ** 2 * s * (s - d2))
        )
        return c_R * delta


@dataclass(frozen=True)
class CollisionBarrier:
    """Compactly supported pairwise collision-avoidance barrier.

    Let ``s = ||delta||^2``, ``a = d_min^2`` and ``b = d_act^2``.  For
    ``a < s < b`` we use

        B_c = weight * (1 - xi)^4 / xi,
        xi = (s - a) / (b - a),

    and set ``B_c = 0`` for ``s >= b``.  The potential diverges as the
    collision boundary is approached, while its value, gradient, and Hessian
    all vanish at the activation distance.  Therefore a non-edge pair can
    enter or leave the collision-neighbor set at ``d_act`` without injecting
    a discontinuity into the virtual velocity or its first derivative.
    """

    collision_distance: float
    activation_distance: float
    weight: float = 0.02

    def __post_init__(self) -> None:
        if self.collision_distance <= 0.0:
            raise ValueError("collision_distance must be positive.")
        if self.activation_distance <= self.collision_distance:
            raise ValueError(
                "activation_distance must exceed collision_distance."
            )
        if self.weight <= 0.0:
            raise ValueError("weight must be positive.")

    def value_and_gradient(self, delta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = self.value_gradient_hessian(delta)
        return value, gradient

    def value_gradient_hessian(
        self, delta: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        delta = np.asarray(delta, dtype=float).reshape(-1)
        s = float(delta @ delta)
        d2 = self.collision_distance**2
        r2 = self.activation_distance**2
        if s <= d2:
            raise FloatingPointError(
                "Collision barrier domain violated: expected ||delta|| > "
                f"{self.collision_distance:.6g}, got "
                f"||delta||={np.sqrt(max(s, 0.0)):.6g}."
            )
        if s >= r2:
            return 0.0, np.zeros_like(delta), np.zeros((delta.size, delta.size))

        span = r2 - d2
        xi = (s - d2) / span

        # f(xi) = (1-xi)^4 / xi
        one_minus = 1.0 - xi
        f = one_minus**4 / xi
        f_prime = -1.0 / xi**2 + 6.0 - 8.0 * xi + 3.0 * xi**2
        f_second = 2.0 / xi**3 - 8.0 + 6.0 * xi

        value = self.weight * f
        dB_ds = self.weight * f_prime / span
        d2B_ds2 = self.weight * f_second / span**2

        gradient = 2.0 * dB_ds * delta
        hessian = (
            2.0 * dB_ds * np.eye(delta.size)
            + 4.0 * d2B_ds2 * np.outer(delta, delta)
        )
        return float(value), gradient, hessian
