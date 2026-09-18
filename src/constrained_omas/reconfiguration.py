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
    distinct. Prospective edges already enter the continuous controller, but
    they do not count toward graph connectivity until the activation guard is
    satisfied and a topology switch is executed.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.active = scenario.initial_active.copy()
        self.staging = np.zeros(scenario.num_robots, dtype=bool)
        self.established_edges: set[Edge] = set(scenario.initial_edges)
        self.current_desired_positions = scenario.desired_positions.copy()
        self.prospective: dict[Edge, ProspectiveEdgeState] = {}
        self.switch_events: list[SwitchEvent] = []
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
        i, j = edge
        distance = float(np.linalg.norm(positions[i] - positions[j]))
        if distance <= self.scenario.collision_distance:
            raise RuntimeError(
                f"Prospective edge {edge} cannot be assigned at t={t:.3f} s: "
                f"d={distance:.3f} <= d_min={self.scenario.collision_distance:.3f}. "
                "The lower collision boundary is not relaxed."
            )
        cfg = self.scenario.reconfiguration
        rho = max(0.0, distance - self.scenario.sensing_distance + cfg.epsilon_rho)
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
        target = (
            request.target_desired_positions.copy()
            if request.target_desired_positions is not None
            else self.current_desired_positions.copy()
        )

        # The desired formation associated with the forthcoming mode must be
        # compatible with every edge that survives the departure. Prospective
        # bridge candidates are checked separately in _bridge_edges().
        for edge in retained:
            self._check_desired_edge(edge, target)

        self._pending_kind = "leave"
        self._pending_robot = robot
        self._pending_target_desired = target
        self._started_leave.add(robot)

        if is_connected(remaining, retained):
            self._pending_leave_phase = "direct"
            return

        bridges = self._bridge_edges(robot, target, positions)
        for edge in bridges:
            self._assign_prospective(
                edge,
                target,
                positions,
                t,
                purpose="bridge",
            )
        self._pending_leave_phase = "forming_bridges"

    def _start_due_request(self, t: float, positions: np.ndarray) -> None:
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
        _, kind, request = min(due, key=lambda item: item[0])
        if kind == "join":
            self._start_join(request, positions, t)  # type: ignore[arg-type]
        else:
            self._start_leave(request, positions, t)  # type: ignore[arg-type]

    def update_relaxation_rates(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        cfg = self.scenario.reconfiguration
        for state in self.prospective.values():
            i, j = state.edge
            delta = positions[i] - positions[j]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                raise RuntimeError("Relative distance vanished on a prospective edge.")
            distance_rate = float(delta @ (velocities[i] - velocities[j]) / distance)
            margin = self.scenario.sensing_distance + state.rho - distance
            state.rho_dot = max(
                -cfg.k_rho * state.rho,
                distance_rate - cfg.lambda_rho * margin,
            )

    def _prospective_ready(self, state: ProspectiveEdgeState, positions: np.ndarray) -> bool:
        cfg = self.scenario.reconfiguration
        i, j = state.edge
        distance = float(np.linalg.norm(positions[i] - positions[j]))
        return (
            state.rho <= cfg.rho_activation_threshold + 1e-12
            and self.scenario.collision_distance + cfg.edge_activation_margin
            <= distance
            <= self.scenario.sensing_distance - cfg.edge_activation_margin
        )

    def _finish_join(self, t: float) -> None:
        assert self._pending_robot is not None
        robot = self._pending_robot
        join_edges = tuple(
            sorted(state.edge for state in self.prospective.values() if state.purpose == "join")
        )
        self.active[robot] = True
        self.staging[robot] = False
        self.established_edges.update(join_edges)
        for edge in join_edges:
            del self.prospective[edge]
        self.switch_events.append(
            SwitchEvent(t, "robot_join", robot, edges_added=join_edges)
        )
        self.last_switch_time = float(t)
        self._pending_kind = None
        self._pending_robot = None

    def _activate_bridges(self, t: float) -> None:
        bridge_edges = tuple(
            sorted(state.edge for state in self.prospective.values() if state.purpose == "bridge")
        )
        self.established_edges.update(bridge_edges)
        for edge in bridge_edges:
            del self.prospective[edge]
        assert self._pending_target_desired is not None
        self.current_desired_positions = self._pending_target_desired.copy()
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
        self.active[robot] = False
        self.staging[robot] = False
        if self._pending_target_desired is not None:
            self.current_desired_positions = self._pending_target_desired.copy()
        self.switch_events.append(
            SwitchEvent(t, "robot_leave", robot, edges_removed=removed)
        )
        self.last_switch_time = float(t)
        self._pending_kind = None
        self._pending_robot = None
        self._pending_leave_phase = None
        self._pending_target_desired = None

    def process(self, t: float, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Process requests, relaxation rates, and admissible topology switches."""
        self._start_due_request(t, positions)
        self.update_relaxation_rates(positions, velocities)

        if self._pending_kind == "join":
            join_states = [s for s in self.prospective.values() if s.purpose == "join"]
            if join_states and all(self._prospective_ready(s, positions) for s in join_states):
                if self._can_switch(t):
                    self._finish_join(t)

        elif self._pending_kind == "leave":
            if self._pending_leave_phase == "forming_bridges":
                bridge_states = [s for s in self.prospective.values() if s.purpose == "bridge"]
                if bridge_states and all(self._prospective_ready(s, positions) for s in bridge_states):
                    if self._can_switch(t):
                        self._activate_bridges(t)
            elif self._pending_leave_phase in ("bridge_active", "direct"):
                if self._can_switch(t):
                    self._finish_leave(t)

        # A switch may have removed all prospective edges; refresh rates only
        # when prospective interactions remain.
        if self.prospective:
            self.update_relaxation_rates(positions, velocities)

    def controller_interactions(self) -> list[EdgeInteraction]:
        interactions: list[EdgeInteraction] = []
        for edge in sorted(self.established_edges):
            interactions.append(
                EdgeInteraction(
                    edge=edge,
                    desired_relative=self._desired_relative(edge, self.current_desired_positions),
                )
            )
        for edge in sorted(self.prospective):
            state = self.prospective[edge]
            interactions.append(
                EdgeInteraction(
                    edge=edge,
                    desired_relative=state.desired_relative,
                    relaxation=state.rho,
                    relaxation_rate=state.rho_dot,
                    prospective=True,
                )
            )
        return interactions

    def integrate_relaxations(self, dt: float, next_positions: np.ndarray) -> None:
        """Euler-step rho and apply a tiny discrete feasibility projection."""
        for state in self.prospective.values():
            state.rho = max(0.0, state.rho + dt * state.rho_dot)
            i, j = state.edge
            next_distance = float(np.linalg.norm(next_positions[i] - next_positions[j]))
            # Continuous-time dynamics preserve s>0. The projection only removes
            # O(dt^2) numerical boundary crossings from the explicit simulation.
            state.rho = max(
                state.rho,
                next_distance - self.scenario.sensing_distance + 1e-9,
            )

    def assert_established_graph_connected(self) -> None:
        vertices = np.flatnonzero(self.active).tolist()
        if not is_connected(vertices, self.established_edges):
            raise RuntimeError("Formation manager produced a disconnected active graph.")
