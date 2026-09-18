from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .barrier import EdgeBarrier
from .graph import Edge, canonical_edge, incidence_matrix


@dataclass(frozen=True)
class ControllerConfig:
    k1: float = 1.0
    k2: float = 1.0
    k3: float = 2.0
    max_virtual_speed: float = 0.8


@dataclass(frozen=True)
class EdgeInteraction:
    """One edge appearing in the distributed controller.

    ``relaxation`` is zero for an established edge and positive for a
    prospective edge. ``relaxation_rate`` is only needed to evaluate the exact
    continuous-time derivative of the virtual velocity.
    """

    edge: Edge
    desired_relative: np.ndarray
    relaxation: float = 0.0
    relaxation_rate: float = 0.0
    prospective: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge", canonical_edge(*self.edge))
        object.__setattr__(
            self,
            "desired_relative",
            np.asarray(self.desired_relative, dtype=float).reshape(-1),
        )
        if self.relaxation < -1e-12:
            raise ValueError("relaxation must be nonnegative.")


class SmoothBacksteppingController:
    """Second-order BLF/backstepping controller used in the paper prototype."""

    def __init__(
        self,
        num_robots: int,
        dimension: int,
        collision_distance: float,
        sensing_distance: float,
        config: ControllerConfig,
    ) -> None:
        self.num_robots = int(num_robots)
        self.dimension = int(dimension)
        self.collision_distance = float(collision_distance)
        self.sensing_distance = float(sensing_distance)
        self.config = config

        if self.num_robots <= 0:
            raise ValueError("num_robots must be positive.")
        if self.dimension not in (2, 3):
            raise ValueError("Only 2-D and 3-D controllers are supported.")
        if self.collision_distance <= 0.0:
            raise ValueError("collision_distance must be positive.")
        if self.sensing_distance <= self.collision_distance:
            raise ValueError("sensing_distance must exceed collision_distance.")

    @classmethod
    def from_desired_offsets(
        cls,
        desired_offsets: np.ndarray,
        collision_distance: float,
        sensing_distance: float,
        config: ControllerConfig,
    ) -> "SmoothBacksteppingController":
        """Compatibility constructor for tests and simple fixed-mode examples."""
        desired_offsets = np.asarray(desired_offsets, dtype=float)
        if desired_offsets.ndim != 2:
            raise ValueError("desired_offsets must have shape (N,dimension).")
        controller = cls(
            num_robots=desired_offsets.shape[0],
            dimension=desired_offsets.shape[1],
            collision_distance=collision_distance,
            sensing_distance=sensing_distance,
            config=config,
        )
        controller._compat_desired_offsets = desired_offsets.copy()
        return controller

    def interactions_from_offsets(
        self,
        edges: Iterable[Edge],
        desired_offsets: np.ndarray | None = None,
    ) -> list[EdgeInteraction]:
        """Build ordinary edge interactions from desired robot positions."""
        if desired_offsets is None:
            desired_offsets = getattr(self, "_compat_desired_offsets", None)
            if desired_offsets is None:
                raise ValueError("desired_offsets must be supplied.")
        desired_offsets = np.asarray(desired_offsets, dtype=float)
        return [
            EdgeInteraction(
                edge=(i, j),
                desired_relative=desired_offsets[i] - desired_offsets[j],
            )
            for i, j in edges
        ]

    def _edge_barrier(self, interaction: EdgeInteraction) -> EdgeBarrier:
        return EdgeBarrier(
            desired_relative=interaction.desired_relative,
            collision_distance=self.collision_distance,
            sensing_distance=self.sensing_distance + interaction.relaxation,
        )

    def edge_quantities(
        self,
        positions: np.ndarray,
        interactions: list[EdgeInteraction],
        *,
        with_hessians: bool = False,
        with_radius_derivatives: bool = False,
    ) -> tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Return aggregate barrier quantities for active/prospective edges."""
        positions = np.asarray(positions, dtype=float)
        edges = [interaction.edge for interaction in interactions]
        E = incidence_matrix(self.num_robots, edges)
        gradients = np.zeros((len(edges), self.dimension), dtype=float)
        hessians = (
            np.zeros((len(edges), self.dimension, self.dimension), dtype=float)
            if with_hessians
            else None
        )
        radius_derivatives = (
            np.zeros((len(edges), self.dimension), dtype=float)
            if with_radius_derivatives
            else None
        )
        total_barrier = 0.0

        for k, interaction in enumerate(interactions):
            i, j = interaction.edge
            delta = positions[i] - positions[j]
            error = delta - interaction.desired_relative
            barrier = self._edge_barrier(interaction)
            if with_hessians:
                value, grad, hessian = barrier.value_gradient_hessian(error)
                assert hessians is not None
                hessians[k] = hessian
            else:
                value, grad = barrier.value_and_gradient(error)
            if with_radius_derivatives:
                assert radius_derivatives is not None
                radius_derivatives[k] = barrier.gradient_upper_radius_derivative(error)
            total_barrier += value
            gradients[k] = grad

        q = E @ gradients if interactions else np.zeros_like(positions)
        return total_barrier, gradients, q, hessians, radius_derivatives

    @staticmethod
    def radial_arctan_saturation(z: np.ndarray, limit: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        norms = np.linalg.norm(z, axis=1)
        out = np.zeros_like(z)
        nonzero = norms > 0.0
        radii = (2.0 * limit / np.pi) * np.arctan(
            np.pi * norms[nonzero] / (2.0 * limit)
        )
        out[nonzero] = radii[:, None] * z[nonzero] / norms[nonzero, None]
        return out

    @staticmethod
    def radial_arctan_saturation_directional_derivative(
        z: np.ndarray,
        z_dot: np.ndarray,
        limit: float,
    ) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z_dot = np.asarray(z_dot, dtype=float)
        if z.shape != z_dot.shape:
            raise ValueError("z and z_dot must have the same shape.")

        derivative = np.zeros_like(z)
        norms = np.linalg.norm(z, axis=1)
        for i, r in enumerate(norms):
            if r <= 1e-12:
                derivative[i] = z_dot[i]
                continue
            unit = z[i] / r
            scaled_radius = np.pi * r / (2.0 * limit)
            rho = (2.0 * limit / np.pi) * np.arctan(scaled_radius)
            rho_prime = 1.0 / (1.0 + scaled_radius**2)
            tangential_gain = rho / r
            radial_component = float(unit @ z_dot[i])
            derivative[i] = (
                tangential_gain * z_dot[i]
                + (rho_prime - tangential_gain) * radial_component * unit
            )
        return derivative

    def virtual_velocity(
        self,
        positions: np.ndarray,
        interactions: list[EdgeInteraction],
        controlled: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        barrier, _, q, _, _ = self.edge_quantities(positions, interactions)
        v_star = -self.radial_arctan_saturation(
            self.config.k1 * q,
            self.config.max_virtual_speed,
        )
        v_star[~controlled] = 0.0
        return v_star, barrier, q

    def virtual_velocity_derivative(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        interactions: list[EdgeInteraction],
        controlled: np.ndarray,
    ) -> np.ndarray:
        """Exact derivative, including the time variation of prospective radii."""
        if not interactions:
            return np.zeros_like(velocities)

        edges = [interaction.edge for interaction in interactions]
        E = incidence_matrix(self.num_robots, edges)
        _, _, q, hessians, radius_derivatives = self.edge_quantities(
            positions,
            interactions,
            with_hessians=True,
            with_radius_derivatives=True,
        )
        assert hessians is not None
        assert radius_derivatives is not None

        edge_velocities = E.T @ velocities
        edge_gradient_derivatives = np.einsum(
            "kij,kj->ki",
            hessians,
            edge_velocities,
        )
        rho_dot = np.asarray(
            [interaction.relaxation_rate for interaction in interactions],
            dtype=float,
        )
        edge_gradient_derivatives += radius_derivatives * rho_dot[:, None]
        q_dot = E @ edge_gradient_derivatives

        z = self.config.k1 * q
        z_dot = self.config.k1 * q_dot
        derivative = -self.radial_arctan_saturation_directional_derivative(
            z,
            z_dot,
            self.config.max_virtual_speed,
        )
        derivative[~controlled] = 0.0
        return derivative

    def control(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        interactions: list[EdgeInteraction],
        controlled: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """Evaluate the distributed backstepping control law.

        Prospective interactions enter exactly like established interactions;
        their only difference is the relaxed upper barrier radius.
        """
        edges = [interaction.edge for interaction in interactions]
        E = incidence_matrix(self.num_robots, edges)
        L = E @ E.T

        v_star, barrier, q = self.virtual_velocity(
            positions,
            interactions,
            controlled,
        )
        v_tilde = np.asarray(velocities, dtype=float) - v_star
        v_tilde[~controlled] = 0.0

        v_star_dot = self.virtual_velocity_derivative(
            positions,
            velocities,
            interactions,
            controlled,
        )

        u = (
            -self.config.k2 * (L @ v_tilde)
            -self.config.k3 * v_tilde
            -q
            + v_star_dot
        )
        u[~controlled] = 0.0

        lyapunov = barrier + 0.5 * float(np.sum(v_tilde[controlled] ** 2))
        return {
            "u": u,
            "v_star": v_star,
            "v_tilde": v_tilde,
            "v_star_dot": v_star_dot,
            "q": q,
            "barrier": barrier,
            "lyapunov": lyapunov,
        }
