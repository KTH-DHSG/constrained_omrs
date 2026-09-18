from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .charging import ChargingAreaConfig
from .graph import Edge, canonical_edge, is_connected


@dataclass(frozen=True)
class ReconfigurationConfig:
    """Parameters of the prospective-edge relaxation mechanism."""

    epsilon_rho: float = 0.10
    k_rho: float = 0.8
    lambda_rho: float = 2.0
    rho_activation_threshold: float = 0.08
    edge_activation_margin: float = 0.05
    min_switch_dwell_time: float = 1.0

    def __post_init__(self) -> None:
        if self.epsilon_rho <= 0.0:
            raise ValueError("epsilon_rho must be positive.")
        if self.k_rho <= 0.0 or self.lambda_rho <= 0.0:
            raise ValueError("k_rho and lambda_rho must be positive.")
        if self.rho_activation_threshold < 0.0:
            raise ValueError("rho_activation_threshold must be nonnegative.")
        if self.edge_activation_margin <= 0.0:
            raise ValueError("edge_activation_margin must be positive.")
        if self.min_switch_dwell_time < 0.0:
            raise ValueError("min_switch_dwell_time must be nonnegative.")


@dataclass(frozen=True)
class JoinRequest:
    robot: int
    time: float
    num_edges: int = 1

    def __post_init__(self) -> None:
        if self.robot < 0:
            raise ValueError("robot index must be nonnegative.")
        if self.time < 0.0:
            raise ValueError("join-request time must be nonnegative.")
        if self.num_edges < 1:
            raise ValueError("A joining robot must be assigned at least one edge.")


@dataclass(frozen=True)
class LeaveRequest:
    robot: int
    time: float
    target_desired_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.robot < 0:
            raise ValueError("robot index must be nonnegative.")
        if self.time < 0.0:
            raise ValueError("leave-request time must be nonnegative.")
        if self.target_desired_positions is not None:
            object.__setattr__(
                self,
                "target_desired_positions",
                np.asarray(self.target_desired_positions, dtype=float),
            )


@dataclass(frozen=True)
class Scenario:
    desired_positions: np.ndarray
    initial_positions: np.ndarray
    initial_velocities: np.ndarray
    initial_active: np.ndarray
    initial_edges: tuple[Edge, ...]
    join_requests: tuple[JoinRequest, ...]
    leave_requests: tuple[LeaveRequest, ...]
    collision_distance: float
    sensing_distance: float
    reconfiguration: ReconfigurationConfig
    duration: float
    dt: float
    charging_area: ChargingAreaConfig | None = None

    def __post_init__(self) -> None:
        desired = np.asarray(self.desired_positions, dtype=float)
        positions = np.asarray(self.initial_positions, dtype=float)
        velocities = np.asarray(self.initial_velocities, dtype=float)
        active = np.asarray(self.initial_active, dtype=bool)

        if desired.ndim != 2:
            raise ValueError("desired_positions must have shape (num_robots, dimension).")
        n, dim = desired.shape
        if dim not in (2, 3):
            raise ValueError("Only 2-D and 3-D scenarios are supported.")
        if positions.shape != desired.shape or velocities.shape != desired.shape:
            raise ValueError("initial positions/velocities must match desired_positions.")
        if active.shape != (n,):
            raise ValueError("initial_active must have shape (num_robots,).")
        if self.collision_distance <= 0.0:
            raise ValueError("collision_distance must be positive.")
        if self.sensing_distance <= self.collision_distance:
            raise ValueError("sensing_distance must exceed collision_distance.")
        if self.duration <= 0.0 or self.dt <= 0.0:
            raise ValueError("duration and dt must be positive.")
        if 2.0 * self.reconfiguration.edge_activation_margin >= (
            self.sensing_distance - self.collision_distance
        ):
            raise ValueError("edge_activation_margin leaves no activation annulus.")

        if self.charging_area is not None:
            if dim != 3:
                raise ValueError("charging_area is currently supported only for 3-D scenarios.")
            if self.charging_area.pad_positions.shape != (n, 3):
                raise ValueError(
                    "charging_area.pad_positions must have shape (num_robots, 3)."
                )

        edges = tuple(canonical_edge(*edge) for edge in self.initial_edges)
        if len(edges) != len(set(edges)):
            raise ValueError("initial_edges must not contain duplicates.")
        active_vertices = set(np.flatnonzero(active).tolist())
        for edge in edges:
            if min(edge) < 0 or max(edge) >= n:
                raise ValueError(f"Invalid initial edge {edge}.")
            if edge[0] not in active_vertices or edge[1] not in active_vertices:
                raise ValueError(f"Initial edge {edge} has an inactive endpoint.")
            d = float(np.linalg.norm(positions[edge[0]] - positions[edge[1]]))
            if not self.collision_distance < d < self.sensing_distance:
                raise ValueError(
                    f"Initial edge {edge} violates the admissible interval: d={d:.3f}."
                )
            desired_d = float(np.linalg.norm(desired[edge[0]] - desired[edge[1]]))
            if not self.collision_distance < desired_d < self.sensing_distance:
                raise ValueError(
                    f"Desired length of initial edge {edge} is outside the nominal annulus."
                )
        if not is_connected(active_vertices, edges):
            raise ValueError("The initial interaction graph must be connected.")

        request_robots = []
        for request in self.join_requests:
            if request.robot >= n:
                raise ValueError("Join request refers to an invalid robot.")
            if active[request.robot]:
                raise ValueError("A robot initially in the team cannot request to join.")
            request_robots.append((request.time, "join", request.robot))
        for request in self.leave_requests:
            if request.robot >= n:
                raise ValueError("Leave request refers to an invalid robot.")
            if request.target_desired_positions is not None:
                if request.target_desired_positions.shape != desired.shape:
                    raise ValueError(
                        "target_desired_positions must match desired_positions shape."
                    )
            request_robots.append((request.time, "leave", request.robot))

        times = [item[0] for item in request_robots]
        if len(times) != len(set(times)):
            raise ValueError(
                "The compact prototype accepts one membership request at a time; "
                "request times must be distinct."
            )

        object.__setattr__(self, "desired_positions", desired)
        object.__setattr__(self, "initial_positions", positions)
        object.__setattr__(self, "initial_velocities", velocities)
        object.__setattr__(self, "initial_active", active)
        object.__setattr__(self, "initial_edges", edges)
        object.__setattr__(self, "join_requests", tuple(sorted(self.join_requests, key=lambda r: r.time)))
        object.__setattr__(self, "leave_requests", tuple(sorted(self.leave_requests, key=lambda r: r.time)))

    @property
    def num_robots(self) -> int:
        return int(self.desired_positions.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.desired_positions.shape[1])

    @property
    def desired_offsets(self) -> np.ndarray:
        """Backward-compatible alias used by some plotting/user code."""
        return self.desired_positions

    @property
    def request_times(self) -> np.ndarray:
        return np.asarray(
            sorted([r.time for r in self.join_requests] + [r.time for r in self.leave_requests]),
            dtype=float,
        )


def _default_2d_data():
    # Initially the four active robots form a chain 0--1--2--3. Robot 4 is
    # parked outside sensing range and requests admission later.
    desired = np.array(
        [
            [-1.5, 0.0],
            [-0.5, 0.0],
            [0.5, 0.0],
            [1.5, 0.0],
            [2.5, 0.0],
        ],
        dtype=float,
    )
    initial = np.array(
        [
            [-1.55, 0.08],
            [-0.48, -0.10],
            [0.55, 0.08],
            [1.48, -0.08],
            [4.15, 0.45],
        ],
        dtype=float,
    )
    velocity = np.array(
        [
            [0.04, 0.00],
            [-0.02, 0.02],
            [0.01, -0.02],
            [-0.03, 0.01],
            [0.00, 0.00],
        ],
        dtype=float,
    )
    target_after_leave = desired.copy()
    # Removal of robot 1 requires bridge (0,2). The old desired 0--2 distance
    # is 2 m and is incompatible with d_max=1.6 m. The forthcoming formation
    # shortens that distance while keeping all retained edges feasible.
    target_after_leave[0] = [-0.70, 0.0]
    target_after_leave[1] = [-0.10, 0.0]  # transitional target before departure
    return desired, initial, velocity, target_after_leave


def _default_3d_data():
    """Spatial seven-robot formation used by the richer 3-D demonstration.

    Robots 0--4 form an initially connected spatial chain. Robots 5 and 6
    subsequently join near opposite ends of the chain, each through two
    prospective links. Robot 2 later leaves; since it is an articulation
    vertex, a make-before-break bridge is formed between the two remaining
    components before the departure is completed.
    """
    desired = np.array(
        [
            [-1.70, -0.55, 1.80],
            [-0.90,  0.30, 2.35],
            [ 0.00, -0.35, 2.90],
            [ 0.90,  0.35, 2.25],
            [ 1.70, -0.50, 2.75],
            [ 1.95,  0.65, 3.15],
            [-1.95,  0.60, 2.75],
        ],
        dtype=float,
    )

    initial = np.array(
        [
            [-1.72, -0.64, 1.75],
            [-0.80,  0.15, 2.50],
            [ 0.10, -0.55, 3.00],
            [ 1.05,  0.25, 2.15],
            [ 1.78, -0.70, 2.95],
            [ 0.40, -2.75, 0.16],
            [-0.40, -2.75, 0.16],
        ],
        dtype=float,
    )

    velocity = np.array(
        [
            [ 0.05,  0.02,  0.00],
            [-0.03,  0.03, -0.02],
            [ 0.02, -0.04,  0.01],
            [-0.02,  0.02,  0.02],
            [ 0.03, -0.02, -0.01],
            [ 0.00,  0.00,  0.00],
            [ 0.00,  0.00,  0.00],
        ],
        dtype=float,
    )

    # Post-departure mode: the two surviving components are compacted slightly
    # before robot 2 is removed. In particular, edge (1,3) becomes the bridge.
    target_after_leave = desired.copy()
    target_after_leave[0] = [-1.35, -0.45, 1.90]
    target_after_leave[1] = [-0.65,  0.25, 2.35]
    target_after_leave[3] = [ 0.65,  0.30, 2.30]
    target_after_leave[4] = [ 1.35, -0.45, 2.75]
    target_after_leave[5] = [ 1.60,  0.55, 3.10]
    target_after_leave[6] = [-1.60,  0.50, 2.70]

    return desired, initial, velocity, target_after_leave


def _default_3d_scenario() -> Scenario:
    desired, initial, velocity, target = _default_3d_data()

    # A compact charging station is placed to one side of the working volume.
    # Robots 5 and 6 start on their pads with motors off; robot 2 later returns
    # to its own pad after the make-before-break departure has completed.
    charging_pads = np.array(
        [
            [-1.20, -3.55, 0.16],
            [-0.40, -3.55, 0.16],
            [ 0.40, -3.55, 0.16],
            [ 1.20, -3.55, 0.16],
            [-1.20, -2.75, 0.16],
            [ 0.40, -2.75, 0.16],
            [-0.40, -2.75, 0.16],
        ],
        dtype=float,
    )
    charging_area = ChargingAreaConfig(
        pad_positions=charging_pads,
        approach_height=0.90,
        approach_radius=0.50,
        maximum_speed=1.30,
        maximum_descent_speed=0.50,
        velocity_gain=2.2,
        maximum_acceleration=3.5,
        landing_position_tolerance=0.10,
        landing_speed_tolerance=0.20,
        platform_padding=0.34,
    )

    return Scenario(
        desired_positions=desired,
        initial_positions=initial,
        initial_velocities=velocity,
        initial_active=np.array([True, True, True, True, True, False, False]),
        initial_edges=((0, 1), (1, 2), (2, 3), (3, 4)),
        join_requests=(
            JoinRequest(robot=5, time=7.0, num_edges=2),
            JoinRequest(robot=6, time=19.0, num_edges=2),
        ),
        leave_requests=(
            LeaveRequest(
                robot=2,
                time=35.0,
                target_desired_positions=target,
            ),
        ),
        collision_distance=0.45,
        sensing_distance=1.65,
        reconfiguration=ReconfigurationConfig(
            epsilon_rho=0.14,
            k_rho=0.75,
            lambda_rho=2.0,
            rho_activation_threshold=0.08,
            edge_activation_margin=0.04,
            min_switch_dwell_time=1.0,
        ),
        duration=52.0,
        dt=0.01,
        charging_area=charging_area,
    )

def default_open_formation_scenario(dimension: int = 2) -> Scenario:
    """Return the default open-formation experiment.

    The 2-D case remains the compact five-robot join/removal example used for
    formulation debugging. The 3-D case is a richer seven-robot spatial
    demonstration with two joining robots, multiple prospective edges, and a
    later make-before-break removal of an articulation robot.
    """
    if dimension == 3:
        return _default_3d_scenario()
    if dimension != 2:
        raise ValueError("dimension must be either 2 or 3.")

    desired, initial, velocity, target = _default_2d_data()
    return Scenario(
        desired_positions=desired,
        initial_positions=initial,
        initial_velocities=velocity,
        initial_active=np.array([True, True, True, True, False]),
        initial_edges=((0, 1), (1, 2), (2, 3)),
        join_requests=(JoinRequest(robot=4, time=6.0, num_edges=1),),
        leave_requests=(
            LeaveRequest(
                robot=1,
                time=18.0,
                target_desired_positions=target,
            ),
        ),
        collision_distance=0.45,
        sensing_distance=1.60,
        reconfiguration=ReconfigurationConfig(
            epsilon_rho=0.12,
            k_rho=0.8,
            lambda_rho=2.0,
            rho_activation_threshold=0.08,
            edge_activation_margin=0.04,
            min_switch_dwell_time=1.0,
        ),
        duration=30.0,
        dt=0.01,
    )
