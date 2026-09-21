from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import csv
import json
from pathlib import Path
from threading import Lock

import numpy as np

from constrained_omrs.graph import active_pairs
from constrained_omrs.simulation import SimulationResult


class GazeboMissionRecorder:
    """Downsample and post-process one physical Gazebo mission.

    The recorder starts at the OMRS mission clock, i.e. after the initial
    service-flight deployment is complete.  This keeps the generated plots
    directly comparable with the pure-Python OMRS experiment while Gazebo
    itself still visualizes takeoff and landing phases live.
    """

    def __init__(self, scenario, quadrotor_config, sample_rate_hz: float) -> None:
        self.scenario = scenario
        self.quadrotor_config = quadrotor_config
        self.sample_period = 1.0 / float(sample_rate_hz)
        self.last_sample_time = -np.inf
        self._lock = Lock()
        self._samples: list[dict] = []
        self._previous_velocity: np.ndarray | None = None
        self._previous_time: float | None = None

    def due(self, scenario_time: float) -> bool:
        return scenario_time - self.last_sample_time >= self.sample_period - 1e-9

    @staticmethod
    def _edge_metrics(manager, positions: np.ndarray, velocities: np.ndarray):
        pos_sq = 0.0
        vel_sq = 0.0
        max_edge_distance = np.nan
        for edge in sorted(manager.established_edges):
            i, j = edge
            desired = manager.desired_relative_for_edge(edge)
            error = positions[i] - positions[j] - desired
            rel_velocity = velocities[i] - velocities[j]
            pos_sq += float(error @ error)
            vel_sq += float(rel_velocity @ rel_velocity)
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            max_edge_distance = distance if np.isnan(max_edge_distance) else max(max_edge_distance, distance)

        pairs = active_pairs(manager.active)
        min_pair_distance = (
            min(float(np.linalg.norm(positions[i] - positions[j])) for i, j in pairs)
            if pairs else np.nan
        )
        return (
            float(np.sqrt(pos_sq)),
            float(np.sqrt(vel_sq)),
            float(min_pair_distance),
            float(max_edge_distance),
        )

    def record(
        self,
        *,
        scenario_time: float,
        manager,
        positions: np.ndarray,
        velocities: np.ndarray,
        rotations: np.ndarray,
        angular_velocities: np.ndarray,
        returning: np.ndarray,
        parked: np.ndarray,
        raw_acceleration_commands: np.ndarray,
        applied_acceleration_commands: np.ndarray,
        virtual_velocities: np.ndarray,
        lyapunov: float,
        attitude_errors: np.ndarray,
        commanded_thrusts: np.ndarray,
        commanded_torques: np.ndarray,
    ) -> None:
        if not self.due(scenario_time):
            return

        positions = np.asarray(positions, dtype=float).copy()
        velocities = np.asarray(velocities, dtype=float).copy()
        if self._previous_velocity is None or self._previous_time is None:
            realized_acceleration = np.zeros_like(velocities)
        else:
            dt = max(float(scenario_time - self._previous_time), 1e-9)
            realized_acceleration = (velocities - self._previous_velocity) / dt
        self._previous_velocity = velocities.copy()
        self._previous_time = float(scenario_time)
        self.last_sample_time = float(scenario_time)

        position_error, velocity_error, min_pair, max_edge = self._edge_metrics(
            manager, positions, velocities
        )
        prospective = {edge: float(state.rho) for edge, state in manager.prospective.items()}
        desired_edges = {
            edge: manager.desired_relative_for_edge(edge).copy()
            for edge in manager.established_edges
        }
        desired_edges.update(
            {
                edge: state.desired_relative.copy()
                for edge, state in manager.prospective.items()
            }
        )
        prospective_margin = {
            edge: float(
                self.scenario.sensing_distance
                + state.rho
                - np.linalg.norm(positions[edge[0]] - positions[edge[1]])
            )
            for edge, state in manager.prospective.items()
        }

        physical_flight = ~np.asarray(parked, dtype=bool)
        sample = {
            "time": float(scenario_time),
            "positions": positions,
            "velocities": velocities,
            "controls": np.asarray(raw_acceleration_commands, dtype=float).copy(),
            "flight_acceleration_commands": np.asarray(applied_acceleration_commands, dtype=float).copy(),
            "realized_accelerations": realized_acceleration,
            "virtual_velocities": np.asarray(virtual_velocities, dtype=float).copy(),
            "lyapunov": float(lyapunov),
            "active": manager.active.copy(),
            "controlled": manager.controlled.copy(),
            "flight_controlled": physical_flight.copy(),
            "returning": np.asarray(returning, dtype=bool).copy(),
            "parked": np.asarray(parked, dtype=bool).copy(),
            "edges": list(sorted(manager.established_edges)),
            "prospective": prospective,
            "prospective_margin": prospective_margin,
            "desired_positions": manager.current_desired_positions.copy(),
            "desired_edges": desired_edges,
            "min_pair_distance": min_pair,
            "max_edge_distance": max_edge,
            "position_error": position_error,
            "velocity_error": velocity_error,
            "rotations": np.asarray(rotations, dtype=float).copy(),
            "angular_velocities": np.asarray(angular_velocities, dtype=float).copy(),
            "attitude_errors": np.asarray(attitude_errors, dtype=float).copy(),
            "commanded_thrusts": np.asarray(commanded_thrusts, dtype=float).copy(),
            "commanded_torques": np.asarray(commanded_torques, dtype=float).copy(),
        }
        with self._lock:
            self._samples.append(sample)

    def build_result(self, switch_events) -> tuple[SimulationResult, object]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            raise RuntimeError("No Gazebo mission samples were recorded.")

        def stack(key):
            return np.stack([sample[key] for sample in samples], axis=0)

        time = np.asarray([sample["time"] for sample in samples], dtype=float)
        sample_dt = float(np.median(np.diff(time))) if len(time) > 1 else self.sample_period
        plot_scenario = replace(
            self.scenario,
            duration=max(float(time[-1]), sample_dt),
            dt=sample_dt,
        )

        result = SimulationResult(
            time=time,
            positions=stack("positions"),
            velocities=stack("velocities"),
            controls=stack("controls"),
            flight_acceleration_commands=stack("flight_acceleration_commands"),
            realized_accelerations=stack("realized_accelerations"),
            virtual_velocities=stack("virtual_velocities"),
            lyapunov=np.asarray([sample["lyapunov"] for sample in samples]),
            active=stack("active").astype(bool),
            controlled=stack("controlled").astype(bool),
            flight_controlled=stack("flight_controlled").astype(bool),
            returning=stack("returning").astype(bool),
            parked=stack("parked").astype(bool),
            edge_history=[sample["edges"] for sample in samples],
            prospective_history=[sample["prospective"] for sample in samples],
            prospective_margin_history=[sample["prospective_margin"] for sample in samples],
            desired_positions=stack("desired_positions"),
            desired_edge_history=[sample["desired_edges"] for sample in samples],
            min_pair_distance=np.asarray([sample["min_pair_distance"] for sample in samples]),
            max_established_edge_distance=np.asarray([sample["max_edge_distance"] for sample in samples]),
            position_edge_error_norm=np.asarray([sample["position_error"] for sample in samples]),
            velocity_edge_error_norm=np.asarray([sample["velocity_error"] for sample in samples]),
            switch_events=tuple(switch_events),
            dynamics_model="quadrotor",
            attitudes=stack("rotations"),
            desired_attitudes=None,
            angular_velocities=stack("angular_velocities"),
            thrusts=stack("commanded_thrusts"),
            commanded_thrusts=stack("commanded_thrusts"),
            torques=stack("commanded_torques"),
            commanded_torques=stack("commanded_torques"),
            attitude_errors=stack("attitude_errors"),
            quadrotor_config=self.quadrotor_config,
        )
        return result, plot_scenario



def save_diagnostic_csvs(result: SimulationResult, scenario, run_dir: Path) -> list[Path]:
    """Save human-readable robot- and edge-level logs for debugging.

    These CSV files complement the lossless NPZ history and are intentionally
    generated for both completed and interrupted / failed runs.
    """
    robot_path = run_dir / "robot_diagnostics.csv"
    robot_fields = [
        "time", "robot", "active", "controlled", "flight_controlled",
        "returning", "parked",
        "px", "py", "pz", "vx", "vy", "vz", "speed",
        "vstar_x", "vstar_y", "vstar_z", "vstar_norm",
        "u_raw_x", "u_raw_y", "u_raw_z", "u_raw_norm",
        "u_applied_x", "u_applied_y", "u_applied_z", "u_applied_norm",
        "a_realized_x", "a_realized_y", "a_realized_z", "a_realized_norm",
        "raw_applied_error_norm", "applied_realized_error_norm",
        "raw_applied_cosine",
        "attitude_error_rad", "thrust_N", "torque_norm_Nm",
        "formation_position_error_norm", "formation_velocity_error_norm",
    ]
    with robot_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=robot_fields)
        writer.writeheader()
        for k, t in enumerate(result.time):
            for robot in range(result.positions.shape[1]):
                p = result.positions[k, robot]
                v = result.velocities[k, robot]
                vs = result.virtual_velocities[k, robot]
                ur = result.controls[k, robot]
                ua = result.flight_acceleration_commands[k, robot]
                ar = result.realized_accelerations[k, robot]
                torque = result.commanded_torques[k, robot]
                raw_applied_error = float(np.linalg.norm(ur - ua))
                applied_realized_error = float(np.linalg.norm(ua - ar))
                denominator = float(np.linalg.norm(ur) * np.linalg.norm(ua))
                raw_applied_cosine = (
                    float(np.dot(ur, ua) / denominator) if denominator > 1e-12 else np.nan
                )
                writer.writerow({
                    "time": float(t), "robot": robot,
                    "active": int(result.active[k, robot]),
                    "controlled": int(result.controlled[k, robot]),
                    "flight_controlled": int(result.flight_controlled[k, robot]),
                    "returning": int(result.returning[k, robot]),
                    "parked": int(result.parked[k, robot]),
                    "px": p[0], "py": p[1], "pz": p[2],
                    "vx": v[0], "vy": v[1], "vz": v[2], "speed": np.linalg.norm(v),
                    "vstar_x": vs[0], "vstar_y": vs[1], "vstar_z": vs[2], "vstar_norm": np.linalg.norm(vs),
                    "u_raw_x": ur[0], "u_raw_y": ur[1], "u_raw_z": ur[2], "u_raw_norm": np.linalg.norm(ur),
                    "u_applied_x": ua[0], "u_applied_y": ua[1], "u_applied_z": ua[2], "u_applied_norm": np.linalg.norm(ua),
                    "a_realized_x": ar[0], "a_realized_y": ar[1], "a_realized_z": ar[2], "a_realized_norm": np.linalg.norm(ar),
                    "raw_applied_error_norm": raw_applied_error,
                    "applied_realized_error_norm": applied_realized_error,
                    "raw_applied_cosine": raw_applied_cosine,
                    "attitude_error_rad": result.attitude_errors[k, robot],
                    "thrust_N": result.commanded_thrusts[k, robot],
                    "torque_norm_Nm": np.linalg.norm(torque),
                    "formation_position_error_norm": result.position_edge_error_norm[k],
                    "formation_velocity_error_norm": result.velocity_edge_error_norm[k],
                })

    edge_path = run_dir / "edge_diagnostics.csv"
    edge_fields = [
        "time", "i", "j", "kind", "distance", "desired_distance",
        "formation_error_norm", "relative_speed", "rho",
        "upper_bound", "upper_margin", "lower_margin",
    ]
    dmin = float(scenario.collision_distance)
    dmax = float(scenario.sensing_distance)
    with edge_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=edge_fields)
        writer.writeheader()
        for k, t in enumerate(result.time):
            established = set(tuple(edge) for edge in result.edge_history[k])
            prospective = {tuple(edge): float(rho) for edge, rho in result.prospective_history[k].items()}
            for edge in sorted(established | set(prospective)):
                i, j = edge
                delta = result.positions[k, i] - result.positions[k, j]
                if result.desired_edge_history is not None:
                    desired = np.asarray(
                        result.desired_edge_history[k].get(
                            edge,
                            result.desired_positions[k, i]
                            - result.desired_positions[k, j],
                        ),
                        dtype=float,
                    )
                else:
                    desired = result.desired_positions[k, i] - result.desired_positions[k, j]
                distance = float(np.linalg.norm(delta))
                rho = prospective.get(edge, 0.0)
                upper = dmax + rho if edge in prospective else dmax
                writer.writerow({
                    "time": float(t), "i": i, "j": j,
                    "kind": "prospective" if edge in prospective else "established",
                    "distance": distance,
                    "desired_distance": float(np.linalg.norm(desired)),
                    "formation_error_norm": float(np.linalg.norm(delta - desired)),
                    "relative_speed": float(np.linalg.norm(result.velocities[k, i] - result.velocities[k, j])),
                    "rho": rho, "upper_bound": upper,
                    "upper_margin": upper - distance,
                    "lower_margin": distance - dmin,
                })
    return [robot_path, edge_path]


def _source_package_outputs_directory(package_name: str = "constrained_omrs") -> Path:
    """Resolve ``<workspace>/src/<package>/outputs`` for a colcon workspace.

    ``get_package_prefix`` points to ``<workspace>/install/<package>`` for a
    source-built ROS package.  Walking back to the workspace keeps generated
    diagnostics in the source package, rather than under ``install/`` where a
    clean rebuild would delete them.  A conservative local fallback is kept for
    non-colcon use.
    """
    try:
        from ament_index_python.packages import get_package_prefix

        prefix = Path(get_package_prefix(package_name)).expanduser().resolve()
        workspace = prefix.parent.parent
        source_package = workspace / "src" / package_name
        if source_package.is_dir():
            return source_package / "outputs"
    except Exception:
        pass

    # Pure-Python / source-tree fallback.  gazebo_results.py lives in
    # <package>/src/constrained_omrs_ros/.
    source_tree = Path(__file__).resolve().parents[2]
    if (source_tree / "package.xml").is_file():
        return source_tree / "outputs"
    return Path.cwd() / "outputs"


def create_run_directory(base_directory: str) -> Path:
    token = str(base_directory).strip()
    if token in ("", "package_outputs", "PACKAGE_OUTPUTS"):
        base = _source_package_outputs_directory()
    else:
        base = Path(token).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    # Avoid a same-second collision if a failed run is restarted immediately.
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    latest = base / "latest_run.txt"
    latest.write_text(str(run_dir) + "\n", encoding="utf-8")
    return run_dir


def save_raw_history(result: SimulationResult, run_dir: Path, completion_summary: dict) -> None:
    np.savez_compressed(
        run_dir / "gazebo_history.npz",
        time=result.time,
        positions=result.positions,
        velocities=result.velocities,
        raw_acceleration_commands=result.controls,
        applied_acceleration_commands=result.flight_acceleration_commands,
        realized_accelerations=result.realized_accelerations,
        virtual_velocities=result.virtual_velocities,
        lyapunov=result.lyapunov,
        min_pair_distance=result.min_pair_distance,
        max_established_edge_distance=result.max_established_edge_distance,
        active=result.active,
        controlled=result.controlled,
        flight_controlled=result.flight_controlled,
        returning=result.returning,
        parked=result.parked,
        desired_positions=result.desired_positions,
        attitudes=result.attitudes,
        angular_velocities=result.angular_velocities,
        attitude_errors=result.attitude_errors,
        commanded_thrusts=result.commanded_thrusts,
        commanded_torques=result.commanded_torques,
        position_edge_error_norm=result.position_edge_error_norm,
        velocity_edge_error_norm=result.velocity_edge_error_norm,
    )
    metadata = dict(completion_summary)
    metadata["switch_events"] = [
        {
            "time": float(event.time),
            "kind": event.kind,
            "robot": event.robot,
            "edges_added": [list(edge) for edge in event.edges_added],
            "edges_removed": [list(edge) for edge in event.edges_removed],
        }
        for event in result.switch_events
    ]
    metadata["files"] = {
        "raw_history": "gazebo_history.npz",
        "robot_diagnostics": "robot_diagnostics.csv",
        "edge_diagnostics": "edge_diagnostics.csv",
        "controller_exceptions": "controller_exceptions.csv",
        "join_transition_log": "join_transition_log.csv",
        "animation_gif": "gazebo_mission.gif",
        "animation_mp4": "gazebo_mission.mp4",
    }
    (run_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def render_run_outputs(
    result: SimulationResult,
    plot_scenario,
    run_dir: Path,
    virtual_speed_limit: float,
    *,
    animation_fps: int,
    animation_stride: int,
) -> list[Path]:
    # Matplotlib is intentionally imported only during post-processing.
    # The ROS / Gazebo controller must not depend on a working plotting stack
    # in order to start or fly the mission.
    from constrained_omrs.plotting import save_animation, save_plots

    paths = save_plots(result, plot_scenario, run_dir, virtual_speed_limit)
    animation_kwargs = dict(
        fps=int(animation_fps),
        stride=max(1, int(animation_stride)),
        tail_seconds=5.0,
        rotate_3d_camera=False,
    )
    paths.append(
        save_animation(
            result,
            plot_scenario,
            run_dir / "gazebo_mission.gif",
            **animation_kwargs,
        )
    )
    try:
        paths.append(
            save_animation(
                result,
                plot_scenario,
                run_dir / "gazebo_mission.mp4",
                **animation_kwargs,
            )
        )
    except RuntimeError:
        # GIF output is always available through Pillow; MP4 is an additional
        # convenience when ffmpeg is installed on the host.
        pass
    return paths


def resolve_saved_run_directory(run_directory: str | Path | None = None) -> Path:
    """Resolve an explicit Gazebo run directory or the package's latest run."""
    if run_directory is not None and str(run_directory).strip():
        run_dir = Path(run_directory).expanduser().resolve()
    else:
        latest = _source_package_outputs_directory() / "latest_run.txt"
        if not latest.is_file():
            raise FileNotFoundError(
                f"Could not find {latest}. Pass a run directory explicitly."
            )
        run_dir = Path(latest.read_text(encoding="utf-8").strip()).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Gazebo run directory does not exist: {run_dir}")
    return run_dir


def load_saved_gazebo_run(
    run_directory: str | Path | None = None,
):
    """Reconstruct the plotting result from a previously saved Gazebo run.

    This loader is intentionally aimed at deterministic post-processing.  It
    combines the lossless numerical history with ``edge_diagnostics.csv`` and
    ``summary.json`` so paper plots and plot animations can be regenerated
    without rerunning Gazebo.
    """
    from constrained_omrs.quadrotor import QuadrotorConfig
    from constrained_omrs.reconfiguration import SwitchEvent
    from constrained_omrs.scenario import default_open_formation_scenario

    run_dir = resolve_saved_run_directory(run_directory)
    history_path = run_dir / "gazebo_history.npz"
    summary_path = run_dir / "summary.json"
    edge_path = run_dir / "edge_diagnostics.csv"
    for required in (history_path, summary_path, edge_path):
        if not required.is_file():
            raise FileNotFoundError(f"Saved Gazebo run is missing: {required}")

    with np.load(history_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    time = np.asarray(arrays["time"], dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("gazebo_history.npz contains an invalid time vector.")
    n_samples = len(time)

    edge_history: list[list[tuple[int, int]]] = [[] for _ in range(n_samples)]
    prospective_history: list[dict[tuple[int, int], float]] = [dict() for _ in range(n_samples)]
    prospective_margin_history: list[dict[tuple[int, int], float]] = [dict() for _ in range(n_samples)]
    inferred_dmin: list[float] = []
    inferred_dmax: list[float] = []

    def sample_index(t: float) -> int:
        index = int(np.clip(np.searchsorted(time, t), 0, n_samples - 1))
        candidates = [index]
        if index > 0:
            candidates.append(index - 1)
        return min(candidates, key=lambda idx: abs(float(time[idx]) - t))

    with edge_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            k = sample_index(float(row["time"]))
            edge = (int(row["i"]), int(row["j"]))
            kind = row["kind"].strip()
            distance = float(row["distance"])
            rho = float(row["rho"])
            upper = float(row["upper_bound"])
            lower_margin = float(row["lower_margin"])
            upper_margin = float(row["upper_margin"])
            inferred_dmin.append(distance - lower_margin)
            inferred_dmax.append(upper - rho)
            if kind == "prospective":
                prospective_history[k][edge] = rho
                prospective_margin_history[k][edge] = upper_margin
            else:
                edge_history[k].append(edge)

    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    switch_events = tuple(
        SwitchEvent(
            time=float(item["time"]),
            kind=item["kind"],
            robot=None if item.get("robot") is None else int(item["robot"]),
            edges_added=tuple(tuple(int(value) for value in edge) for edge in item.get("edges_added", [])),
            edges_removed=tuple(tuple(int(value) for value in edge) for edge in item.get("edges_removed", [])),
        )
        for item in summary.get("switch_events", [])
    )

    sample_dt = float(np.median(np.diff(time)))
    scenario = default_open_formation_scenario(3)
    dmin = float(np.median(inferred_dmin)) if inferred_dmin else scenario.collision_distance
    dmax = float(np.median(inferred_dmax)) if inferred_dmax else scenario.sensing_distance
    plot_scenario = replace(
        scenario,
        collision_distance=dmin,
        sensing_distance=dmax,
        duration=max(float(time[-1]), sample_dt),
        dt=sample_dt,
    )

    def arr(name: str, *, dtype=float):
        return np.asarray(arrays[name], dtype=dtype)

    result = SimulationResult(
        time=time,
        positions=arr("positions"),
        velocities=arr("velocities"),
        controls=arr("raw_acceleration_commands"),
        flight_acceleration_commands=arr("applied_acceleration_commands"),
        realized_accelerations=arr("realized_accelerations"),
        virtual_velocities=arr("virtual_velocities"),
        lyapunov=arr("lyapunov"),
        active=arr("active", dtype=bool),
        controlled=arr("controlled", dtype=bool),
        flight_controlled=arr("flight_controlled", dtype=bool),
        returning=arr("returning", dtype=bool),
        parked=arr("parked", dtype=bool),
        edge_history=edge_history,
        prospective_history=prospective_history,
        prospective_margin_history=prospective_margin_history,
        desired_positions=arr("desired_positions"),
        min_pair_distance=arr("min_pair_distance"),
        max_established_edge_distance=arr("max_established_edge_distance"),
        position_edge_error_norm=arr("position_edge_error_norm"),
        velocity_edge_error_norm=arr("velocity_edge_error_norm"),
        switch_events=switch_events,
        dynamics_model="quadrotor",
        attitudes=arr("attitudes") if "attitudes" in arrays else None,
        desired_attitudes=None,
        angular_velocities=arr("angular_velocities") if "angular_velocities" in arrays else None,
        thrusts=arr("commanded_thrusts") if "commanded_thrusts" in arrays else None,
        commanded_thrusts=arr("commanded_thrusts") if "commanded_thrusts" in arrays else None,
        torques=arr("commanded_torques") if "commanded_torques" in arrays else None,
        commanded_torques=arr("commanded_torques") if "commanded_torques" in arrays else None,
        attitude_errors=arr("attitude_errors") if "attitude_errors" in arrays else None,
        desired_edge_history=None,
        quadrotor_config=QuadrotorConfig(),
    )
    return result, plot_scenario, run_dir
