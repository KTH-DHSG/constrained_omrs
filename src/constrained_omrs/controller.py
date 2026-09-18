from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .barrier import CollisionBarrier, EdgeBarrier
from .graph import Edge, canonical_edge, incidence_matrix


@dataclass(frozen=True)
class ControllerConfig:
    k1: float = 1.0
    k2: float = 1.0
    k3: float = 2.0
    max_virtual_speed: float = 0.8
    collision_weight: float = 0.02
    # Smooth radial bounds on the dissipative backstepping terms.  Saturating
    # the *damping* contributions, rather than clipping the finished control
    # vector, preserves their sign in the standard Lyapunov derivative for
    # ordinary undirected edges.  The feedforward / BLF-gradient contribution
    # remains untouched.
    max_edge_damping: float = 0.60
    max_local_damping: float = 1.00

    def __post_init__(self) -> None:
        for name in (
            "k1", "k2", "k3", "max_virtual_speed", "collision_weight",
            "max_edge_damping", "max_local_damping",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")


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
    responders: tuple[int, ...] | None = None
    response_weights: tuple[float, float] | None = None
    response_weight_rates: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge", canonical_edge(*self.edge))
        object.__setattr__(
            self,
            "desired_relative",
            np.asarray(self.desired_relative, dtype=float).reshape(-1),
        )
        if self.relaxation < -1e-12:
            raise ValueError("relaxation must be nonnegative.")
        if self.responders is not None and self.response_weights is not None:
            raise ValueError("Use either responders or response_weights, not both.")
        if self.responders is not None:
            responders = tuple(int(robot) for robot in self.responders)
            if not responders:
                raise ValueError("responders must be nonempty when provided.")
            if any(robot not in self.edge for robot in responders):
                raise ValueError("Every responder must be an endpoint of the interaction edge.")
            object.__setattr__(self, "responders", responders)
        if self.response_weights is not None:
            weights = tuple(float(value) for value in self.response_weights)
            if len(weights) != 2 or any(value < -1e-12 or value > 1.0 + 1e-12 for value in weights):
                raise ValueError("response_weights must contain two values in [0,1].")
            object.__setattr__(self, "response_weights", weights)
        if self.response_weight_rates is not None:
            if self.response_weights is None:
                raise ValueError("response_weight_rates require response_weights.")
            rates = tuple(float(value) for value in self.response_weight_rates)
            if len(rates) != 2:
                raise ValueError("response_weight_rates must contain two values.")
            object.__setattr__(self, "response_weight_rates", rates)


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
        if self.config.collision_weight <= 0.0:
            raise ValueError("collision_weight must be positive.")

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


    def _interaction_operators(
        self, interactions: list[EdgeInteraction]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return relative, response, and response-rate incidence operators.

        ``E`` is the ordinary incidence matrix used to form relative edge
        states. ``R`` determines how strongly each endpoint reacts to each
        interaction. For an ordinary undirected edge, ``R == E``. Before a
        joining robot is admitted, only the candidate endpoint reacts. During
        the optional post-admission bumpless handover, the old-team endpoint
        is smoothly ramped from zero to full response. ``R_dot`` is included
        explicitly so the analytical derivative of the saturated virtual
        velocity remains correct during that handover.
        """
        edges = [interaction.edge for interaction in interactions]
        E = incidence_matrix(self.num_robots, edges)
        R = E.copy()
        R_dot = np.zeros_like(E)
        for k, interaction in enumerate(interactions):
            if interaction.response_weights is not None:
                i, j = interaction.edge
                wi, wj = interaction.response_weights
                dwi, dwj = interaction.response_weight_rates or (0.0, 0.0)
                R[:, k] = 0.0
                R_dot[:, k] = 0.0
                R[i, k] = wi
                R[j, k] = -wj
                R_dot[i, k] = dwi
                R_dot[j, k] = -dwj
                continue
            if interaction.responders is None:
                continue
            R[:, k] = 0.0
            i, j = interaction.edge
            for robot in interaction.responders:
                R[robot, k] = 1.0 if robot == i else -1.0
        return E, R, R_dot

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
        _, R, _ = self._interaction_operators(interactions)
        gradients = np.zeros((len(interactions), self.dimension), dtype=float)
        hessians = (
            np.zeros((len(interactions), self.dimension, self.dimension), dtype=float)
            if with_hessians
            else None
        )
        radius_derivatives = (
            np.zeros((len(interactions), self.dimension), dtype=float)
            if with_radius_derivatives
            else None
        )
        total_barrier = 0.0

        for k, interaction in enumerate(interactions):
            i, j = interaction.edge
            delta = positions[i] - positions[j]
            error = delta - interaction.desired_relative
            barrier = self._edge_barrier(interaction)
            try:
                if with_hessians:
                    value, grad, hessian = barrier.value_gradient_hessian(error)
                    assert hessians is not None
                    hessians[k] = hessian
                else:
                    value, grad = barrier.value_and_gradient(error)
                if with_radius_derivatives:
                    assert radius_derivatives is not None
                    radius_derivatives[k] = barrier.gradient_upper_radius_derivative(error)
            except FloatingPointError as exc:
                distance = float(np.linalg.norm(delta))
                desired_distance = float(np.linalg.norm(interaction.desired_relative))
                raise FloatingPointError(
                    f"{exc} interaction_edge={interaction.edge}, "
                    f"prospective={interaction.prospective}, rho={interaction.relaxation:.6g}, "
                    f"d={distance:.6g}, d_des={desired_distance:.6g}."
                ) from exc
            total_barrier += value
            gradients[k] = grad

        q = R @ gradients if interactions else np.zeros_like(positions)
        return total_barrier, gradients, q, hessians, radius_derivatives

    def nearby_collision_pairs(
        self,
        positions: np.ndarray,
        participants: np.ndarray,
        excluded_edges: Iterable[Edge] = (),
    ) -> list[Edge]:
        """Return non-edge robot pairs inside the collision sensing radius.

        Connectivity / prospective edges are normally passed through
        ``excluded_edges`` because their ordinary edge BLF already contains the
        lower collision barrier.  Every other participating pair closer than
        ``sensing_distance`` receives the compact collision-only barrier.
        """
        positions = np.asarray(positions, dtype=float)
        participants = np.asarray(participants, dtype=bool).reshape(self.num_robots)
        excluded = {canonical_edge(*edge) for edge in excluded_edges}
        pairs: list[Edge] = []
        for i in range(self.num_robots):
            if not participants[i]:
                continue
            for j in range(i + 1, self.num_robots):
                if not participants[j]:
                    continue
                edge = (i, j)
                if edge in excluded:
                    continue
                if np.linalg.norm(positions[i] - positions[j]) < self.sensing_distance:
                    pairs.append(edge)
        return pairs

    def collision_quantities(
        self,
        positions: np.ndarray,
        collision_pairs: Iterable[Edge],
        *,
        with_hessians: bool = False,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return aggregate quantities for collision-only proximity pairs."""
        positions = np.asarray(positions, dtype=float)
        pairs = [canonical_edge(*edge) for edge in collision_pairs]
        E = incidence_matrix(self.num_robots, pairs)
        gradients = np.zeros((len(pairs), self.dimension), dtype=float)
        hessians = (
            np.zeros((len(pairs), self.dimension, self.dimension), dtype=float)
            if with_hessians
            else None
        )
        barrier = CollisionBarrier(
            collision_distance=self.collision_distance,
            activation_distance=self.sensing_distance,
            weight=self.config.collision_weight,
        )
        total = 0.0
        for k, (i, j) in enumerate(pairs):
            delta = positions[i] - positions[j]
            if with_hessians:
                value, gradient, hessian = barrier.value_gradient_hessian(delta)
                assert hessians is not None
                hessians[k] = hessian
            else:
                value, gradient = barrier.value_and_gradient(delta)
            total += value
            gradients[k] = gradient
        q = E @ gradients if pairs else np.zeros_like(positions)
        return total, gradients, q, hessians

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
        collision_pairs: Iterable[Edge] = (),
    ) -> tuple[np.ndarray, float, np.ndarray]:
        edge_barrier, _, q_edge, _, _ = self.edge_quantities(positions, interactions)
        collision_barrier, _, q_collision, _ = self.collision_quantities(
            positions, collision_pairs
        )
        q = q_edge + q_collision
        v_star = -self.radial_arctan_saturation(
            self.config.k1 * q,
            self.config.max_virtual_speed,
        )
        v_star[~controlled] = 0.0
        return v_star, edge_barrier + collision_barrier, q

    def virtual_velocity_derivative(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        interactions: list[EdgeInteraction],
        controlled: np.ndarray,
        collision_pairs: Iterable[Edge] = (),
    ) -> np.ndarray:
        """Exact derivative of the saturated virtual velocity.

        Prospective-edge radius dynamics and the compact collision-only barrier
        Hessians are both included.  Because the collision potential and its
        first two derivatives vanish at ``d_max``, changing the proximity-pair
        set at that boundary does not inject a jump into this expression.
        """
        collision_pairs = [canonical_edge(*edge) for edge in collision_pairs]
        if not interactions and not collision_pairs:
            return np.zeros_like(velocities)

        E, R, R_dot = self._interaction_operators(interactions)
        _, gradients, q, hessians, radius_derivatives = self.edge_quantities(
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
        q_dot = R @ edge_gradient_derivatives if interactions else np.zeros_like(velocities)
        if interactions:
            # q = R g. A smooth topology handover makes R time-varying, so
            # q_dot contains the additional R_dot g term. Omitting this term
            # would reintroduce an artificial jump in v_star_dot precisely at
            # the transition we are trying to make bumpless.
            q_dot = q_dot + R_dot @ gradients

        _, _, q_collision, collision_hessians = self.collision_quantities(
            positions, collision_pairs, with_hessians=True
        )
        if collision_pairs:
            assert collision_hessians is not None
            E_collision = incidence_matrix(self.num_robots, collision_pairs)
            collision_edge_velocities = E_collision.T @ velocities
            collision_gradient_derivatives = np.einsum(
                "kij,kj->ki", collision_hessians, collision_edge_velocities
            )
            q_dot = q_dot + E_collision @ collision_gradient_derivatives

        q = q + q_collision
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
        collision_pairs: Iterable[Edge] = (),
    ) -> dict[str, np.ndarray | float]:
        """Evaluate the distributed backstepping control law.

        Established and bridge-prospective interactions are symmetric.
        A pre-admission joining interaction may specify a single responder, in
        which case only the candidate robot receives that edge's prospective
        formation/connectivity action until the admission switch.
        """
        E, R, _ = self._interaction_operators(interactions)
        collision_pairs = [canonical_edge(*edge) for edge in collision_pairs]
        edge_barrier, _, q_edge, _, _ = self.edge_quantities(positions, interactions)
        collision_barrier, _, q_collision, _ = self.collision_quantities(
            positions, collision_pairs
        )
        q = q_edge + q_collision
        v_star = -self.radial_arctan_saturation(
            self.config.k1 * q, self.config.max_virtual_speed
        )
        v_star[~controlled] = 0.0
        barrier = edge_barrier + collision_barrier
        v_tilde = np.asarray(velocities, dtype=float) - v_star
        v_tilde[~controlled] = 0.0

        v_star_dot = self.virtual_velocity_derivative(
            positions,
            velocities,
            interactions,
            controlled,
            collision_pairs,
        )

        # Smoothly saturate only the dissipative terms. For an ordinary
        # undirected interaction R == E, hence the edge contribution is
        #
        #   - E S_e(k2 E^T v_tilde),
        #
        # and its contribution to V_dot is
        #
        #   -(E^T v_tilde)^T S_e(k2 E^T v_tilde) <= 0
        #
        # because the radial arctan map is odd and direction preserving. The
        # same argument applies robot-wise to the local k3 damping. This is
        # intentionally different from clipping the completed acceleration
        # vector, which would destroy the cancellation structure of the
        # backstepping derivation. Candidate-only join interactions are a
        # supervisory pre-admission mode outside the main undirected-graph
        # Lyapunov claim, but use the same smooth implementation.
        if interactions:
            edge_velocity_error = E.T @ v_tilde
            edge_damping = self.radial_arctan_saturation(
                self.config.k2 * edge_velocity_error,
                self.config.max_edge_damping,
            )
            distributed_damping = R @ edge_damping
        else:
            edge_velocity_error = np.zeros((0, self.dimension), dtype=float)
            edge_damping = np.zeros_like(edge_velocity_error)
            distributed_damping = np.zeros_like(v_tilde)

        local_damping = self.radial_arctan_saturation(
            self.config.k3 * v_tilde,
            self.config.max_local_damping,
        )
        local_damping[~controlled] = 0.0
        distributed_damping[~controlled] = 0.0

        u = -distributed_damping - local_damping - q + v_star_dot
        u[~controlled] = 0.0

        lyapunov = barrier + 0.5 * float(np.sum(v_tilde[controlled] ** 2))
        return {
            "u": u,
            "edge_velocity_error": edge_velocity_error,
            "edge_damping": edge_damping,
            "distributed_damping": distributed_damping,
            "local_damping": local_damping,
            "v_star": v_star,
            "v_tilde": v_tilde,
            "v_star_dot": v_star_dot,
            "q": q,
            "q_edge": q_edge,
            "q_collision": q_collision,
            "edge_barrier": edge_barrier,
            "collision_barrier": collision_barrier,
            "collision_pairs": collision_pairs,
            "barrier": barrier,
            "lyapunov": lyapunov,
        }
