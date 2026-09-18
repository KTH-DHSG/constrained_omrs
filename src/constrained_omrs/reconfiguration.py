from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .controller import EdgeInteraction
from .graph import Edge, canonical_edge, connected_components, is_connected
from .scenario import JoinRequest, LeaveRequest, Scenario


@dataclass
class ProspectiveEdgeState:
    edge: Edge
    desired_relative: np.ndarray
    rho: float
    assigned_at: float
    purpose: Literal["join", "bridge"]
    rho_dot: float = 0.0


@dataclass
class JoinHandoverState:
    edge: Edge
    candidate: int
    activated_at: float
    duration: float


@dataclass(frozen=True)
class SwitchEvent:
    time: float
    kind: Literal["robot_join", "bridge_activation", "robot_leave"]
    robot: int | None
    edges_added: tuple[Edge, ...] = ()
    edges_removed: tuple[Edge, ...] = ()


class FormationManager:
    """Supervisory open-team reconfiguration mechanism.

    The manager deliberately keeps *established* and *prospective* edges
    distinct. Prospective bridge edges enter the continuous controller
    symmetrically.  A joining prospective edge is pre-admission: only the
    candidate robot responds to it until the admission switch.  Prospective
    edges never count toward graph connectivity until activation.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.active = scenario.initial_active.copy()
        self.staging = np.zeros(scenario.num_robots, dtype=bool)
        self.established_edges: set[Edge] = set(scenario.initial_edges)
        self.current_desired_positions = scenario.desired_positions.copy()
        self.prospective: dict[Edge, ProspectiveEdgeState] = {}
        # Established bridge edges may temporarily carry an edge-specific
        # desired relative displacement during the make-before-break phase.
        # This avoids changing the desired formation of unrelated retained
        # edges at bridge activation. The override is removed when the
        # departing robot actually leaves and the minimally modified
        # post-departure node formation becomes active.
        self.established_desired_overrides: dict[Edge, np.ndarray] = {}
        self.join_handovers: dict[Edge, JoinHandoverState] = {}
        self.switch_events: list[SwitchEvent] = []
        self.current_time = 0.0
        self.last_switch_time = -np.inf

        self._started_join: set[int] = set()
        self._started_leave: set[int] = set()
        self._pending_kind: Literal["join", "leave"] | None = None
        self._pending_robot: int | None = None
        self._pending_leave_phase: Literal["forming_bridges", "bridge_active", "direct"] | None = None
        self._pending_target_desired: np.ndarray | None = None

    @property
    def controlled(self) -> np.ndarray:
        return self.active | self.staging

    def _can_switch(self, t: float) -> bool:
        dwell = self.scenario.reconfiguration.min_switch_dwell_time
        return t - self.last_switch_time >= dwell - 0.5 * self.scenario.dt

    def _desired_relative(self, edge: Edge, desired_positions: np.ndarray) -> np.ndarray:
        i, j = edge
        return desired_positions[i] - desired_positions[j]

    def desired_relative_for_edge(self, edge: Edge) -> np.ndarray:
        """Return the desired relative displacement currently used by an edge.

        Most established edges are induced by ``current_desired_positions``.
        A newly activated departure bridge is the only exception: during the
        short make-before-break intermediate mode it keeps the same desired
        relative displacement it had while prospective. This makes bridge
        activation bumpless and postpones the node-level formation update
        until the departing robot is actually removed.
        """
        edge = canonical_edge(*edge)
        override = self.established_desired_overrides.get(edge)
        if override is not None:
            return override
        return self._desired_relative(edge, self.current_desired_positions)

    def _check_desired_edge(self, edge: Edge, desired_positions: np.ndarray) -> None:
        desired_distance = float(
            np.linalg.norm(self._desired_relative(edge, desired_positions))
        )
        if not (
            self.scenario.collision_distance
            < desired_distance
            < self.scenario.sensing_distance
        ):
            raise RuntimeError(
                f"Desired distance of edge {edge} is {desired_distance:.3f} m, "
                "outside the nominal BLF annulus. The formation manager must "
                "select a compatible desired formation before adding this edge."
            )

    def _prospective_target_is_activation_compatible(
        self,
        edge: Edge,
        desired_positions: np.ndarray,
    ) -> bool:
        """Whether the edge target lies inside the protected activation annulus.

        The relaxation law preserves ``edge_activation_margin`` from the moving
        upper boundary.  Therefore a prospective edge can reach ``rho = 0``
        only if its desired geometry is itself compatible with that protected
        nominal annulus.  Checking this explicitly prevents a prospective edge
        from remaining orange forever because the supervisor selected an
        impossible target for the configured physical margin.
        """
        desired_distance = float(
            np.linalg.norm(self._desired_relative(edge, desired_positions))
        )
        margin = self.scenario.reconfiguration.edge_activation_margin
        return (
            self.scenario.collision_distance + margin
            < desired_distance
            < self.scenario.sensing_distance - margin
        )

    def _assign_prospective(
        self,
        edge: Edge,
        desired_positions: np.ndarray,
        positions: np.ndarray,
        t: float,
        purpose: Literal["join", "bridge"],
    ) -> None:
        edge = canonical_edge(*edge)
        if edge in self.established_edges or edge in self.prospective:
            return
        self._check_desired_edge(edge, desired_positions)
        if not self._prospective_target_is_activation_compatible(edge, desired_positions):
            desired_distance = float(
                np.linalg.norm(self._desired_relative(edge, desired_positions))
            )
            margin = self.scenario.reconfiguration.edge_activation_margin
            raise RuntimeError(
                f"Prospective edge {edge} has desired distance {desired_distance:.3f} m, "
                f"which is incompatible with the protected activation margin "
                f"{margin:.3f} m. Select another edge or minimally adjust the "
                "desired formation before assigning this prospective edge."
            )
        i, j = edge
        distance = float(np.linalg.norm(positions[i] - positions[j]))
        if distance <= self.scenario.collision_distance:
            raise RuntimeError(
                f"Prospective edge {edge} cannot be assigned at t={t:.3f} s: "
                f"d={distance:.3f} <= d_min={self.scenario.collision_distance:.3f}. "
                "The lower collision boundary is not relaxed."
            )
        cfg = self.scenario.reconfiguration
        # Initialize the relaxed radius with a deliberate interior buffer.
        # The BLF singularity remains at d_max + rho, while the relaxation
        # dynamics protect an additional edge_activation_margin from that
        # boundary. Therefore the protected margin at assignment is at least
        # rho_initial_margin, rather than merely an infinitesimal epsilon.
        rho = max(
            0.0,
            distance
            - self.scenario.sensing_distance
            + cfg.edge_activation_margin
            + cfg.rho_initial_margin,
        )
        self.prospective[edge] = ProspectiveEdgeState(
            edge=edge,
            desired_relative=self._desired_relative(edge, desired_positions).copy(),
            rho=float(rho),
            assigned_at=float(t),
            purpose=purpose,
        )

    def _start_join(self, request: JoinRequest, positions: np.ndarray, t: float) -> None:
        robot = request.robot
        if self._pending_kind is not None:
            return
        if self.active[robot]:
            raise RuntimeError(f"Robot {robot} is already active and cannot join again.")

        active_ids = np.flatnonzero(self.active)
        candidates: list[tuple[float, int]] = []
        for j in active_ids:
            edge = canonical_edge(robot, int(j))
            desired_distance = float(
                np.linalg.norm(
                    self.current_desired_positions[robot]
                    - self.current_desired_positions[int(j)]
                )
            )
            if not (
                self.scenario.collision_distance
                < desired_distance
                < self.scenario.sensing_distance
            ):
                continue
            if not self._prospective_target_is_activation_compatible(
                edge, self.current_desired_positions
            ):
                continue
            distance = float(np.linalg.norm(positions[robot] - positions[int(j)]))
            if distance > self.scenario.collision_distance:
                candidates.append((distance, int(j)))

        if len(candidates) < request.num_edges:
            raise RuntimeError(
                f"Robot {robot} has only {len(candidates)} feasible prospective "
                f"neighbors, but {request.num_edges} were requested."
            )
        candidates.sort()
        self.staging[robot] = True
        for _, j in candidates[: request.num_edges]:
            self._assign_prospective(
                (robot, j),
                self.current_desired_positions,
                positions,
                t,
                purpose="join",
            )
        self._pending_kind = "join"
        self._pending_robot = robot
        self._started_join.add(robot)

    def _remaining_components_for_departure(
        self,
        departing_robot: int,
    ) -> tuple[set[int], set[Edge], list[set[int]]]:
        remaining = set(np.flatnonzero(self.active).tolist()) - {departing_robot}
        retained = {
            edge
            for edge in self.established_edges
            if departing_robot not in edge
        }
        components = connected_components(remaining, retained)
        return remaining, retained, components

    def _project_bridge_relative_to_interior(
        self,
        desired_relative: np.ndarray,
        physical_relative: np.ndarray,
    ) -> np.ndarray:
        """Project a bridge target into a comfortably interior nominal annulus.

        The minimum-change departure strategy translates whole connected
        components rigidly. Hence only the relative displacement between
        components needs to be modified. We preserve its direction whenever
        possible and change only its norm, by the smallest amount required to
        lie inside an annulus with one additional activation-margin buffer.

        If the current desired relative vector is degenerate, the measured
        physical relative vector supplies the direction instead.
        """
        cfg = self.scenario.reconfiguration
        dmin = float(self.scenario.collision_distance)
        dmax = float(self.scenario.sensing_distance)
        margin = float(cfg.edge_activation_margin)

        available = max(dmax - dmin - 2.0 * margin, 0.0)
        # A bridge is formed while the departing robot and its old incident
        # edges are still active, so the intermediate graph may contain a
        # temporarily inconsistent cycle. Aim appreciably inside the nominal
        # annulus (four activation margins when geometry permits) so the
        # prospective bridge can reach rho=0 despite that transient conflict.
        extra = min(3.0 * margin, 0.45 * available)
        lower = dmin + margin + extra
        upper = dmax - margin - extra
        if not lower < upper:
            # This should be prevented by Scenario validation, but keep a
            # conservative fallback for unusual parameterizations.
            lower = dmin + margin
            upper = dmax - margin

        desired_relative = np.asarray(desired_relative, dtype=float)
        norm = float(np.linalg.norm(desired_relative))
        if norm > 1e-12:
            direction = desired_relative / norm
        else:
            physical_relative = np.asarray(physical_relative, dtype=float)
            physical_norm = float(np.linalg.norm(physical_relative))
            if physical_norm > 1e-12:
                direction = physical_relative / physical_norm
            else:
                direction = np.zeros_like(desired_relative)
                direction[0] = 1.0
            norm = 0.0

        target_norm = float(np.clip(norm, lower, upper))
        return direction * target_norm

    def _minimal_bridge_plan(
        self,
        departing_robot: int,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, list[Edge]]:
        """Construct bridge edges with minimum rigid component translation.

        Removing an articulation robot may split the surviving desired
        formation into multiple connected components. Instead of replacing the
        complete desired formation, we translate connected components rigidly.
        Therefore every retained intra-component desired relative displacement
        is preserved *exactly*. Only the relative placement of components is
        changed, and only by the amount required to make the selected bridge
        target compatible with the protected nominal annulus.

        For more than two components a greedy minimum-displacement spanning
        construction is used. When two already assembled component groups are
        connected, the required relative correction is distributed between the
        groups so as to minimize the sum of squared node translations.
        """
        remaining, retained, components = self._remaining_components_for_departure(
            departing_robot
        )
        if len(components) <= 1:
            return self.current_desired_positions.copy(), []

        target = self.current_desired_positions.copy()

        # Each group is a set of original connected-component indices. Entire
        # groups are translated together, so previously selected bridge targets
        # and all retained intra-component desired relations remain unchanged.
        groups: list[set[int]] = [{idx} for idx in range(len(components))]
        bridges: list[Edge] = []

        def group_vertices(group: set[int]) -> set[int]:
            out: set[int] = set()
            for idx in group:
                out.update(components[idx])
            return out

        while len(groups) > 1:
            best = None
            for ga in range(len(groups)):
                va = sorted(group_vertices(groups[ga]))
                for gb in range(ga + 1, len(groups)):
                    vb = sorted(group_vertices(groups[gb]))
                    na = len(va)
                    nb = len(vb)
                    for i in va:
                        for j in vb:
                            edge = canonical_edge(i, j)
                            desired_relative = target[i] - target[j]
                            physical_relative = positions[i] - positions[j]
                            bridge_relative = self._project_bridge_relative_to_interior(
                                desired_relative,
                                physical_relative,
                            )
                            correction = bridge_relative - desired_relative

                            # Minimum weighted translation cost subject to
                            # t_A - t_B = correction.
                            weighted_cost = (
                                (na * nb) / float(na + nb)
                            ) * float(correction @ correction)
                            physical_distance = float(np.linalg.norm(physical_relative))
                            candidate = (
                                weighted_cost,
                                max(0.0, physical_distance - self.scenario.sensing_distance),
                                physical_distance,
                                edge,
                                ga,
                                gb,
                                correction,
                            )
                            if best is None or candidate[:4] < best[:4]:
                                best = candidate

            if best is None:
                raise RuntimeError(
                    "Unable to construct a minimum-change bridge plan for the "
                    "post-departure components."
                )

            _, _, _, edge, ga, gb, correction = best
            va = sorted(group_vertices(groups[ga]))
            vb = sorted(group_vertices(groups[gb]))
            na = len(va)
            nb = len(vb)

            # Minimum-norm distribution of the relative correction over all
            # nodes in the two component groups.
            shift_a = (nb / float(na + nb)) * correction
            shift_b = -(na / float(na + nb)) * correction
            target[va] += shift_a
            target[vb] += shift_b

            self._check_desired_edge(edge, target)
            if not self._prospective_target_is_activation_compatible(edge, target):
                raise RuntimeError(
                    f"Minimum-change bridge construction produced an activation-"
                    f"incompatible target for edge {edge}."
                )
            bridges.append(edge)

            merged = groups[ga] | groups[gb]
            groups = [
                group
                for idx, group in enumerate(groups)
                if idx not in (ga, gb)
            ]
            groups.append(merged)

        # The defining invariant of the construction: every edge that survives
        # the departure keeps exactly the same desired relative displacement.
        for edge in retained:
            before = self._desired_relative(edge, self.current_desired_positions)
            after = self._desired_relative(edge, target)
            if not np.allclose(before, after, atol=1e-12, rtol=0.0):
                raise RuntimeError(
                    f"Minimum-change bridge plan altered retained desired edge {edge}."
                )

        # All selected bridges must be compatible with the final target.
        for edge in bridges:
            self._check_desired_edge(edge, target)
            if not self._prospective_target_is_activation_compatible(edge, target):
                raise RuntimeError(
                    f"Bridge {edge} is incompatible with the protected activation "
                    "annulus after minimum-change target construction."
                )

        return target, bridges

    def _bridge_edges(
        self,
        departing_robot: int,
        target_desired: np.ndarray,
        positions: np.ndarray,
    ) -> list[Edge]:
        remaining = set(np.flatnonzero(self.active).tolist()) - {departing_robot}
        retained = {
            edge
            for edge in self.established_edges
            if departing_robot not in edge
        }
        components = connected_components(remaining, retained)
        if len(components) <= 1:
            return []

        # Candidate pairs are ranked by how far they currently lie outside the
        # nominal sensing range, matching the simple supervisory cost discussed
        # in the paper development.
        component_of = {
            vertex: component_index
            for component_index, component in enumerate(components)
            for vertex in component
        }
        candidates: list[tuple[float, Edge, int, int]] = []
        vertices = sorted(remaining)
        for a, i in enumerate(vertices):
            for j in vertices[a + 1 :]:
                ci, cj = component_of[i], component_of[j]
                if ci == cj:
                    continue
                edge = canonical_edge(i, j)
                desired_distance = float(np.linalg.norm(target_desired[i] - target_desired[j]))
                if not (
                    self.scenario.collision_distance
                    < desired_distance
                    < self.scenario.sensing_distance
                ):
                    continue
                if not self._prospective_target_is_activation_compatible(
                    edge, target_desired
                ):
                    continue
                current_distance = float(np.linalg.norm(positions[i] - positions[j]))
                cost = max(0.0, current_distance - self.scenario.sensing_distance)
                candidates.append((cost, edge, ci, cj))

        candidates.sort(key=lambda item: (item[0], item[1]))
        parent = list(range(len(components)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        selected: list[Edge] = []
        for _, edge, ci, cj in candidates:
            if union(ci, cj):
                selected.append(edge)
                if len(selected) == len(components) - 1:
                    break
        if len(selected) != len(components) - 1:
            raise RuntimeError(
                "No set of prospective bridging edges is compatible with the "
                "desired post-departure formation."
            )
        return selected

    def _start_leave(self, request: LeaveRequest, positions: np.ndarray, t: float) -> None:
        robot = request.robot
        if self._pending_kind is not None:
            return
        if not self.active[robot]:
            raise RuntimeError(f"Robot {robot} is not active and cannot leave.")

        remaining = set(np.flatnonzero(self.active).tolist()) - {robot}
        retained = {edge for edge in self.established_edges if robot not in edge}

        # Minimal desired-formation change policy. A departure that does not
        # require a new bridge leaves the desired formation completely
        # unchanged. If a bridge is needed, connected components of the
        # surviving graph are translated rigidly by the minimum amount needed
        # to make a bridge target feasible. This preserves every retained
        # desired relative displacement exactly. The optional supervisory
        # target is kept only as a final fallback for unusual geometries.
        current_target = self.current_desired_positions.copy()
        self._pending_kind = "leave"
        self._pending_robot = robot
        self._started_leave.add(robot)

        if is_connected(remaining, retained):
            self._pending_target_desired = current_target
            self._pending_leave_phase = "direct"
            return

        try:
            target, bridges = self._minimal_bridge_plan(robot, positions)
        except RuntimeError:
            if request.target_desired_positions is None:
                raise
            target = request.target_desired_positions.copy()
            # Even for an explicit fallback target, never silently alter
            # retained surviving edges. A target that does so is incompatible
            # with the minimal-change policy and must be rejected.
            for edge in retained:
                before = self._desired_relative(edge, current_target)
                after = self._desired_relative(edge, target)
                if not np.allclose(before, after, atol=1e-12, rtol=0.0):
                    raise RuntimeError(
                        f"Fallback departure target alters retained desired edge {edge}; "
                        "provide a component-rigid target instead."
                    )
                self._check_desired_edge(edge, target)
            bridges = self._bridge_edges(robot, target, positions)

        self._pending_target_desired = target
        for edge in bridges:
            self._assign_prospective(
                edge,
                target,
                positions,
                t,
                purpose="bridge",
            )
        self._pending_leave_phase = "forming_bridges"

    def _start_due_request(
        self,
        t: float,
        positions: np.ndarray,
        join_ready: np.ndarray | None = None,
    ) -> None:
        if self._pending_kind is not None:
            return
        due: list[tuple[float, str, object]] = []
        for request in self.scenario.join_requests:
            if request.robot not in self._started_join and request.time <= t + 0.5 * self.scenario.dt:
                due.append((request.time, "join", request))
        for request in self.scenario.leave_requests:
            if request.robot not in self._started_leave and request.time <= t + 0.5 * self.scenario.dt:
                due.append((request.time, "leave", request))
        if not due:
            return

        # Requests are processed in chronological order.  The optional
        # ``join_ready`` gate is used only by the Gazebo service layer: a
        # joining drone may first spool its motors and climb vertically clear
        # of the charging station before prospective edges are assigned.  The
        # pure-Python simulation omits this argument and therefore preserves
        # the original behavior.
        _, kind, request = min(due, key=lambda item: item[0])
        if kind == "join":
            join_request = request  # type: ignore[assignment]
            if join_ready is not None and not bool(join_ready[join_request.robot]):
                return
            self._start_join(join_request, positions, t)
        else:
            self._start_leave(request, positions, t)  # type: ignore[arg-type]

    def update_relaxation_rates(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Project a constant nominal contraction onto the admissible rate set.

        The protected relaxed margin is

            s = d_max + rho - d - epsilon_a,

        where ``epsilon_a = edge_activation_margin``.  Requiring
        ``s_dot >= -lambda_rho s`` gives

            rho_dot >= d_dot - lambda_rho s.

        The nominal rate is the constant ``-rho_contraction_rate``.  Hence,
        away from the safety projection, rho reaches zero in finite time. At
        rho=0 the non-negativity constraint is also projected explicitly; rho
        can reopen only if needed to preserve the protected margin.
        """
        cfg = self.scenario.reconfiguration
        for state in self.prospective.values():
            i, j = state.edge
            delta = positions[i] - positions[j]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                raise RuntimeError("Relative distance vanished on a prospective edge.")
            distance_rate = float(delta @ (velocities[i] - velocities[j]) / distance)
            protected_margin = (
                self.scenario.sensing_distance
                + state.rho
                - distance
                - cfg.edge_activation_margin
            )
            nominal_rate = -cfg.rho_contraction_rate if state.rho > 1e-12 else 0.0
            state.rho_dot = max(
                nominal_rate,
                distance_rate - cfg.lambda_rho * protected_margin,
            )

    def _prospective_ready(
        self,
        state: ProspectiveEdgeState,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> bool:
        """Check whether a prospective edge is ready for nominal activation.

        Activation occurs at rho=0 rather than below an arbitrary positive
        threshold.  The protected-margin relaxation law already guarantees a
        fixed interior distance from the nominal upper boundary at rho=0.  We
        additionally require a lower-bound margin and a modest outward radial
        speed, which is the component of relative motion that could immediately
        violate the nominal connectivity constraint after the switch.
        """
        cfg = self.scenario.reconfiguration
        i, j = state.edge
        delta = positions[i] - positions[j]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            return False
        distance_rate = float(delta @ (velocities[i] - velocities[j]) / distance)
        formation_error = float(np.linalg.norm(delta - state.desired_relative))

        # A joining edge is intentionally kept prospective for a finite
        # interval in the physical realization.  This is important even when
        # rho is already zero at assignment: otherwise a candidate that has
        # merely entered the nominal distance annulus can be admitted in the
        # very same manager update, so the candidate-only prospective
        # controller is never actually applied.  The optional error guard is
        # deliberately moderate (not a convergence test); it only prevents a
        # topology switch with a very large relative-vector mismatch.
        join_guard = True
        if state.purpose == "join":
            prospective_age = self.current_time - state.assigned_at
            join_guard = (
                prospective_age
                >= cfg.join_prospective_min_duration - 0.5 * self.scenario.dt
                and formation_error <= cfg.join_activation_error_tolerance
            )

        return (
            state.rho <= 1e-9
            and self.scenario.collision_distance + cfg.edge_activation_margin
            <= distance
            <= self.scenario.sensing_distance - cfg.edge_activation_margin + 1e-9
            and distance_rate <= cfg.edge_activation_outward_rate_tolerance
            and join_guard
        )

    def _finish_join(self, t: float) -> None:
        assert self._pending_robot is not None
        robot = self._pending_robot
        join_edges = tuple(
            sorted(state.edge for state in self.prospective.values() if state.purpose == "join")
        )

        # A join must never reconfigure the desired formation of robots that
        # are already in the team. The candidate is admitted into the existing
        # formation; retained desired relative displacements are invariant.
        desired_before = self.current_desired_positions.copy()

        self.active[robot] = True
        self.staging[robot] = False
        self.established_edges.update(join_edges)
        duration = self.scenario.reconfiguration.join_handover_duration
        for edge in join_edges:
            del self.prospective[edge]
            if duration > 0.0:
                self.join_handovers[edge] = JoinHandoverState(
                    edge=edge, candidate=robot, activated_at=float(t), duration=float(duration)
                )

        if not np.array_equal(self.current_desired_positions, desired_before):
            raise RuntimeError("Joining a robot modified the existing desired formation.")

        self.switch_events.append(
            SwitchEvent(t, "robot_join", robot, edges_added=join_edges)
        )
        self.last_switch_time = float(t)
        self._pending_kind = None
        self._pending_robot = None

    def _activate_bridges(self, t: float) -> None:
        bridge_states = [
            state for state in self.prospective.values() if state.purpose == "bridge"
        ]
        bridge_edges = tuple(sorted(state.edge for state in bridge_states))

        # Make bridge activation itself bumpless. The prospective bridge is
        # already acting symmetrically with rho=0 at this instant, so the same
        # desired relative displacement is retained as an edge-specific
        # established-edge override. Crucially, the node-level desired
        # formation is *not* changed yet: all pre-existing established edges,
        # including those incident to the departing robot, keep exactly the
        # same targets during the intermediate make-before-break mode.
        for state in bridge_states:
            self.established_edges.add(state.edge)
            self.established_desired_overrides[state.edge] = (
                state.desired_relative.copy()
            )
        for edge in bridge_edges:
            del self.prospective[edge]

        self.switch_events.append(
            SwitchEvent(
                t,
                "bridge_activation",
                self._pending_robot,
                edges_added=bridge_edges,
            )
        )
        self.last_switch_time = float(t)
        self._pending_leave_phase = "bridge_active"

    def _finish_leave(self, t: float) -> None:
        assert self._pending_robot is not None
        robot = self._pending_robot
        removed = tuple(sorted(edge for edge in self.established_edges if robot in edge))
        self.established_edges = {edge for edge in self.established_edges if robot not in edge}
        for edge in removed:
            self.established_desired_overrides.pop(edge, None)
        self.active[robot] = False
        self.staging[robot] = False

        if self._pending_target_desired is not None:
            target = self._pending_target_desired.copy()

            # Every surviving override must coincide with the target induced by
            # the new post-departure node formation. This guarantees that the
            # bridge target is continuous across the robot-removal switch.
            for edge, override in list(self.established_desired_overrides.items()):
                if edge not in self.established_edges:
                    self.established_desired_overrides.pop(edge, None)
                    continue
                induced = self._desired_relative(edge, target)
                if not np.allclose(induced, override, atol=1e-10, rtol=0.0):
                    raise RuntimeError(
                        f"Post-departure target is inconsistent with active bridge "
                        f"override on edge {edge}."
                    )

            # Verify once more that all pre-existing surviving edges were
            # preserved by the minimal component-rigid modification.
            for edge in self.established_edges:
                if edge in self.established_desired_overrides:
                    continue
                before = self._desired_relative(edge, self.current_desired_positions)
                after = self._desired_relative(edge, target)
                if not np.allclose(before, after, atol=1e-10, rtol=0.0):
                    raise RuntimeError(
                        f"Departure target unexpectedly changes retained edge {edge}."
                    )

            self.current_desired_positions = target
            # The bridge override is now represented exactly by the node-level
            # post-departure formation and is no longer needed.
            self.established_desired_overrides.clear()

        self.switch_events.append(
            SwitchEvent(t, "robot_leave", robot, edges_removed=removed)
        )
        self.last_switch_time = float(t)
        self._pending_kind = None
        self._pending_robot = None
        self._pending_leave_phase = None
        self._pending_target_desired = None

    def process(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        join_ready: np.ndarray | None = None,
    ) -> None:
        """Process requests, relaxation rates, and admissible topology switches.

        ``join_ready`` is an optional supervisory admission gate.  It is kept
        outside the theoretical reconfiguration mechanism and is used by the
        Gazebo integration to complete a physical takeoff phase before a robot
        is assigned prospective interaction edges.
        """
        self.current_time = float(t)
        if join_ready is not None:
            join_ready = np.asarray(join_ready, dtype=bool).reshape(
                self.scenario.num_robots
            )
        # Retire completed response ramps. The edge remains established; only
        # the temporary asymmetric handover state disappears.
        for edge, state in list(self.join_handovers.items()):
            if t >= state.activated_at + state.duration - 0.5 * self.scenario.dt:
                del self.join_handovers[edge]
        self._start_due_request(t, positions, join_ready)
        self.update_relaxation_rates(positions, velocities)

        if self._pending_kind == "join":
            join_states = [s for s in self.prospective.values() if s.purpose == "join"]
            if join_states and all(
                self._prospective_ready(s, positions, velocities) for s in join_states
            ):
                if self._can_switch(t):
                    self._finish_join(t)

        elif self._pending_kind == "leave":
            if self._pending_leave_phase == "forming_bridges":
                bridge_states = [s for s in self.prospective.values() if s.purpose == "bridge"]
                if bridge_states and all(
                    self._prospective_ready(s, positions, velocities) for s in bridge_states
                ):
                    if self._can_switch(t):
                        self._activate_bridges(t)
            elif self._pending_leave_phase in ("bridge_active", "direct"):
                if self._can_switch(t):
                    self._finish_leave(t)

        # A switch may have removed all prospective edges; refresh rates only
        # when prospective interactions remain.
        if self.prospective:
            self.update_relaxation_rates(positions, velocities)

    @staticmethod
    def _smoothstep01(tau: float) -> tuple[float, float]:
        """C2 smootherstep and derivative with respect to normalized time.

        The quintic profile has zero first and second derivatives at both
        endpoints.  It is used only by the physical/supervisory join handover;
        the nominal OMRS controller itself is unchanged outside that interval.
        """
        tau = float(np.clip(tau, 0.0, 1.0))
        alpha = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
        dalpha_dtau = 30.0 * tau**2 * (1.0 - tau) ** 2
        return alpha, dalpha_dtau

    def handover_progress(self, edge: Edge) -> tuple[float, float] | None:
        edge = canonical_edge(*edge)
        state = self.join_handovers.get(edge)
        if state is None:
            return None
        if state.duration <= 0.0:
            return (1.0, 0.0)
        tau = (self.current_time - state.activated_at) / state.duration
        alpha, dalpha_dtau = self._smoothstep01(tau)
        return alpha, dalpha_dtau / state.duration

    def controller_interactions(
        self,
        handover_mode: Literal["ramped", "pre", "full"] = "ramped",
    ) -> list[EdgeInteraction]:
        """Return the interactions seen by the continuous controller.

        ``ramped`` preserves the original in-controller response interpolation
        used by the pure-Python prototype.  ``pre`` reconstructs the exact
        candidate-only control structure that existed immediately before a
        join switch, whereas ``full`` returns the ordinary post-switch
        undirected interactions.  The Gazebo integration uses ``pre`` and
        ``full`` to perform a bumpless *whole-controller* homotopy; this is
        more robust than interpolating the response matrix inside the
        backstepping law itself.
        """
        if handover_mode not in ("ramped", "pre", "full"):
            raise ValueError("handover_mode must be 'ramped', 'pre', or 'full'.")

        interactions: list[EdgeInteraction] = []
        for edge in sorted(self.established_edges):
            responders = None
            response_weights = None
            response_weight_rates = None
            handover = self.join_handovers.get(edge)
            if handover is not None:
                if handover_mode == "pre":
                    responders = (handover.candidate,)
                elif handover_mode == "ramped":
                    alpha, alpha_dot = self.handover_progress(edge) or (1.0, 0.0)
                    i, j = edge
                    if handover.candidate == i:
                        response_weights = (1.0, alpha)
                        response_weight_rates = (0.0, alpha_dot)
                    elif handover.candidate == j:
                        response_weights = (alpha, 1.0)
                        response_weight_rates = (alpha_dot, 0.0)
                    else:
                        raise RuntimeError(
                            "Join handover candidate is not an edge endpoint."
                        )
                # handover_mode == "full" deliberately leaves the interaction
                # as an ordinary symmetric established edge.
            interactions.append(
                EdgeInteraction(
                    edge=edge,
                    desired_relative=self.desired_relative_for_edge(edge).copy(),
                    responders=responders,
                    response_weights=response_weights,
                    response_weight_rates=response_weight_rates,
                )
            )

        for edge in sorted(self.prospective):
            state = self.prospective[edge]
            responders = None
            if state.purpose == "join":
                # Before admission, the candidate is not part of the current
                # undirected OMRS mode. Only the joining robot reacts to the
                # prospective formation/connectivity edge.
                if self._pending_kind != "join" or self._pending_robot is None:
                    raise RuntimeError(
                        "Join prospective edge has no pending candidate robot."
                    )
                responders = (self._pending_robot,)
            interactions.append(
                EdgeInteraction(
                    edge=edge,
                    desired_relative=state.desired_relative,
                    relaxation=state.rho,
                    relaxation_rate=state.rho_dot,
                    prospective=True,
                    responders=responders,
                )
            )
        return interactions

    def integrate_relaxations(self, dt: float, next_positions: np.ndarray) -> None:
        """Euler-step rho and enforce nonnegativity / protected-margin feasibility."""
        cfg = self.scenario.reconfiguration
        for state in self.prospective.values():
            state.rho = max(0.0, state.rho + dt * state.rho_dot)
            i, j = state.edge
            next_distance = float(np.linalg.norm(next_positions[i] - next_positions[j]))
            # Continuous-time dynamics preserve the protected margin.  This
            # projection only removes discretization error and, importantly,
            # clamps rho exactly to zero when the linear contraction reaches it.
            minimum_rho = (
                next_distance
                - self.scenario.sensing_distance
                + cfg.edge_activation_margin
                + 1e-9
            )
            state.rho = max(state.rho, minimum_rho, 0.0)
            if state.rho < 1e-10:
                state.rho = 0.0

    def assert_established_graph_connected(self) -> None:
        vertices = np.flatnonzero(self.active).tolist()
        if not is_connected(vertices, self.established_edges):
            raise RuntimeError("Formation manager produced a disconnected active graph.")
