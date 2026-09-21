from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .graph import Edge
from .scenario import Scenario
from .simulation import SimulationResult


def _set_3d_equal_box(ax, points: np.ndarray, padding: float = 0.5) -> None:
    """Choose tight 3-D limits while preserving equal physical data scale."""
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = 0.5 * (mins + maxs)
    half_spans = np.maximum(0.5 * (maxs - mins) + padding, padding)
    ax.set_xlim(centers[0] - half_spans[0], centers[0] + half_spans[0])
    ax.set_ylim(centers[1] - half_spans[1], centers[1] + half_spans[1])
    ax.set_zlim(centers[2] - half_spans[2], centers[2] + half_spans[2])
    # Box dimensions proportional to axis spans keep one metre visually
    # comparable along x, y, and z without forcing a large empty cube.
    ax.set_box_aspect(tuple(2.0 * half_spans))


def _add_switch_markers(ax, result: SimulationResult) -> None:
    labels_seen: set[str] = set()
    styles = {
        "robot_join": ("--", "robot join"),
        "bridge_activation": (":", "bridge activation"),
        "robot_leave": ("-.", "robot leave"),
    }
    for event in result.switch_events:
        linestyle, label = styles[event.kind]
        display = label if label not in labels_seen else None
        labels_seen.add(label)
        ax.axvline(
            event.time,
            linestyle=linestyle,
            linewidth=1.0,
            alpha=0.45,
            color="0.4",
            label=display,
        )


def _edge_union(result: SimulationResult) -> list[Edge]:
    edges = {edge for history in result.edge_history for edge in history}
    edges.update(edge for history in result.prospective_history for edge in history)
    return sorted(edges)


def _edge_status_masks(
    result: SimulationResult,
    edge: Edge,
) -> tuple[np.ndarray, np.ndarray]:
    established = np.asarray(
        [edge in history for history in result.edge_history],
        dtype=bool,
    )
    prospective = np.asarray(
        [edge in history for history in result.prospective_history],
        dtype=bool,
    )
    return established, prospective


def _edge_distance(result: SimulationResult, edge: Edge) -> np.ndarray:
    i, j = edge
    return np.linalg.norm(result.positions[:, i] - result.positions[:, j], axis=1)






def _quadrotor_local_geometry(config):
    """Return body-frame wireframe pieces at the configured physical scale."""
    motors = np.asarray(config.rotor_positions_body, dtype=float)
    arms = [motors[[0, 2]], motors[[1, 3]]]

    r = config.body_radius
    h = 0.5 * config.body_height
    top = np.array(
        [
            [ r,  0.65 * r,  h],
            [-r,  0.65 * r,  h],
            [-r, -0.65 * r,  h],
            [ r, -0.65 * r,  h],
            [ r,  0.65 * r,  h],
        ],
        dtype=float,
    )
    bottom = top.copy()
    bottom[:, 2] = -h
    verticals = [
        np.vstack((top[index], bottom[index]))
        for index in range(4)
    ]
    leg_xy = 0.52 * r
    leg_top_z = -h
    leg_bottom_z = -h - config.landing_leg_length
    legs = [
        np.array([[sx * leg_xy, sy * leg_xy, leg_top_z],
                  [sx * leg_xy, sy * leg_xy, leg_bottom_z]], dtype=float)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    body = _join_wireframe_pieces([top, bottom, *verticals, *legs])

    theta = np.linspace(0.0, 2.0 * np.pi, 21)
    propellers = []
    for motor in motors:
        circle = np.column_stack(
            (
                config.propeller_radius * np.cos(theta),
                config.propeller_radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )
        propellers.append(circle + motor)
    return arms, body, propellers


def _join_wireframe_pieces(pieces):
    rows = []
    for index, piece in enumerate(pieces):
        if index:
            rows.append(np.full((1, 3), np.nan))
        rows.append(np.asarray(piece, dtype=float))
    return np.vstack(rows) if rows else np.empty((0, 3), dtype=float)


def _create_quadrotor_artists(ax, color, config):
    # Two artists per vehicle keeps the animation fast while retaining a clear
    # quadrotor silhouette: a thicker structural frame and thinner propeller
    # rings, both rendered at the configured physical dimensions.
    structure = ax.plot([], [], [], color=color, linewidth=2.4)[0]
    rotors = ax.plot([], [], [], color=color, linewidth=1.05)[0]
    return [structure, rotors]


def _set_quadrotor_artists(artists, position, rotation, config, *, visible, active, opacity=None):
    for artist in artists:
        artist.set_visible(bool(visible))
    if not visible:
        return

    arms, body, propellers = _quadrotor_local_geometry(config)
    local_structure = _join_wireframe_pieces([*arms, body])
    local_rotors = _join_wireframe_pieces(propellers)
    alpha = float(opacity) if opacity is not None else (0.95 if active else 0.42)
    for artist, local_points in zip(artists, (local_structure, local_rotors)):
        world = local_points @ rotation.T
        finite = np.isfinite(world[:, 0])
        world[finite] += position
        artist.set_data_3d(world[:, 0], world[:, 1], world[:, 2])
        artist.set_alpha(alpha)


def _draw_charging_area(ax, scenario: Scenario) -> None:
    """Draw the charging platform and one landing pad per vehicle."""
    if scenario.charging_area is None or scenario.dimension != 3:
        return
    pads = scenario.charging_area.pad_positions
    padding = scenario.charging_area.platform_padding
    xmin, ymin = np.min(pads[:, :2], axis=0) - padding
    xmax, ymax = np.max(pads[:, :2], axis=0) + padding
    z = 0.025
    vertices = [[
        (xmin, ymin, z),
        (xmax, ymin, z),
        (xmax, ymax, z),
        (xmin, ymax, z),
    ]]
    platform = Poly3DCollection(
        vertices,
        facecolor="0.82",
        edgecolor="0.45",
        linewidth=0.8,
        alpha=0.26,
    )
    ax.add_collection3d(platform)

    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    pad_radius = 0.26
    for i, pad in enumerate(pads):
        ax.plot(
            pad[0] + pad_radius * np.cos(theta),
            pad[1] + pad_radius * np.sin(theta),
            np.full_like(theta, z + 0.003),
            color="0.45",
            linewidth=0.9,
            alpha=0.65,
        )
        ax.text(
            pad[0],
            pad[1],
            z + 0.01,
            f"C{i + 1}",
            fontsize=6.5,
            color="0.35",
            ha="center",
            va="center",
        )
    ax.text(
        0.5 * (xmin + xmax),
        ymin - 0.07,
        z,
        "charging area",
        fontsize=8.0,
        color="0.30",
        ha="center",
        va="top",
    )


def _aligned_desired_positions(result: SimulationResult, k: int) -> np.ndarray:
    """Return a translationally aligned representative of the desired formation.

    The controller regulates desired *relative* displacements, so the desired
    formation is defined only up to a common translation.  For visualization we
    remove this gauge freedom by translating the desired formation so that the
    centroid of the currently active team coincides with the centroid of the
    corresponding robot positions.  The same translation is then applied to
    staging robots, so their displayed targets remain consistent with the active
    formation they are approaching.
    """
    desired = np.asarray(result.desired_positions[k], dtype=float).copy()
    anchor = np.asarray(result.active[k], dtype=bool)
    if not np.any(anchor):
        anchor = np.asarray(result.controlled[k], dtype=bool)
    if not np.any(anchor):
        return desired

    translation = (
        np.mean(result.positions[k, anchor], axis=0)
        - np.mean(desired[anchor], axis=0)
    )
    return desired + translation


def _save_static_figure(fig, path: Path) -> Path:
    """Save one paper figure in raster and vector form."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Static plot path must use a .png suffix.")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _trajectory_plot(result: SimulationResult, scenario: Scenario, output_dir: Path) -> Path:
    n = scenario.num_robots
    visible = result.flight_controlled | result.parked
    cmap = plt.get_cmap("tab10")

    if scenario.dimension == 2:
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        for i in range(n):
            ids = np.flatnonzero(visible[:, i])
            if ids.size == 0:
                continue
            p = result.positions[ids, i]
            color = cmap(i % 10)
            ax.plot(p[:, 0], p[:, 1], color=color, label=f"robot {i + 1}")
            ax.plot(p[0, 0], p[0, 1], "o", color=color)
            ax.plot(p[-1, 0], p[-1, 1], "x", color=color)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
    else:
        fig = plt.figure(figsize=(8.8, 7.4))
        ax = fig.add_subplot(111, projection="3d")
        _draw_charging_area(ax, scenario)
        for i in range(n):
            ids = np.flatnonzero(visible[:, i])
            if ids.size == 0:
                continue
            p = result.positions[ids, i]
            color = cmap(i % 10)
            ax.plot(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                color=color,
                linewidth=1.6,
                label=f"robot {i + 1}",
            )
            ax.scatter(*p[0], marker="o", color=color, s=28)
            ax.scatter(*p[-1], marker="x", color=color, s=34)

        # Show the final established interaction geometry as a spatial snapshot.
        final_edges = set(result.edge_history[-1])
        for i, j in final_edges:
            pi = result.positions[-1, i]
            pj = result.positions[-1, j]
            ax.plot(
                [pi[0], pj[0]],
                [pi[1], pj[1]],
                [pi[2], pj[2]],
                color="tab:green",
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
            )

        final_active = result.active[-1]
        final_desired = _aligned_desired_positions(result, -1)
        desired_ids = np.flatnonzero(final_active)
        if desired_ids.size:
            ax.scatter(
                final_desired[desired_ids, 0],
                final_desired[desired_ids, 1],
                final_desired[desired_ids, 2],
                marker="+",
                color="0.35",
                s=42,
                alpha=0.55,
                label="final desired positions (translation-aligned)",
            )

        if result.attitudes is not None and result.quadrotor_config is not None:
            final_visible = result.flight_controlled[-1] | result.parked[-1]
            for i in np.flatnonzero(final_visible):
                artists = _create_quadrotor_artists(
                    ax, cmap(i % 10), result.quadrotor_config
                )
                if result.active[-1, i]:
                    opacity = 0.95
                elif result.returning[-1, i]:
                    opacity = 0.78
                elif result.staging[-1, i]:
                    opacity = 0.70
                else:
                    opacity = 0.38
                _set_quadrotor_artists(
                    artists,
                    result.positions[-1, i],
                    result.attitudes[-1, i],
                    result.quadrotor_config,
                    visible=True,
                    active=bool(result.active[-1, i]),
                    opacity=opacity,
                )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        _set_3d_equal_box(ax, result.positions[visible], padding=0.6)
        ax.view_init(elev=24.0, azim=-58.0)

    ax.set_title("Robot trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    path = output_dir / "trajectories.png"
    _save_static_figure(fig, path)
    return path


def _edge_distance_plot(
    result: SimulationResult,
    scenario: Scenario,
    output_dir: Path,
) -> Path:
    """Plot every controller edge distance, with status encoded by line style.

    A prospective edge is shown dashed while its relaxed interaction is being
    established. The same curve switches to a solid line when the edge becomes
    established. Once an edge is removed, its curve stops.
    """
    edges = _edge_union(result)
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    cmap = plt.get_cmap("tab10")
    edge_handles: list[Line2D] = []

    relaxed_bound_handles: list[Line2D] = []

    for index, edge in enumerate(edges):
        color = cmap(index % 10)
        distance = _edge_distance(result, edge)
        established, prospective = _edge_status_masks(result, edge)

        ax.plot(
            result.time,
            np.where(prospective, distance, np.nan),
            linestyle="--",
            linewidth=1.8,
            color=color,
        )
        ax.plot(
            result.time,
            np.where(established, distance, np.nan),
            linestyle="-",
            linewidth=1.9,
            color=color,
        )
        edge_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.9,
                label=rf"$d_{{{edge[0] + 1},{edge[1] + 1}}}$",
            )
        )

        # For every edge that is prospective at some point, show its actual
        # relaxed upper boundary d_max + rho_k using the same edge color.  The
        # curve is drawn only during the prospective phase and therefore ends
        # exactly when the edge is promoted to the established graph.
        rho = np.asarray(
            [history.get(edge, np.nan) for history in result.prospective_history],
            dtype=float,
        )
        if np.any(np.isfinite(rho)):
            relaxed_bound = scenario.sensing_distance + rho
            ax.plot(
                result.time,
                relaxed_bound,
                linestyle="-.",
                linewidth=1.45,
                color=color,
                alpha=0.9,
            )
            relaxed_bound_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=1.45,
                    linestyle="-.",
                    label=rf"$d_{{\max}}+\rho_{{{edge[0] + 1},{edge[1] + 1}}}$",
                )
            )

    lower = ax.axhline(
        scenario.collision_distance,
        linestyle=":",
        linewidth=1.5,
        color="0.15",
        label=r"$d_{\min}$",
    )
    upper = ax.axhline(
        scenario.sensing_distance,
        linestyle=":",
        linewidth=1.5,
        color="0.45",
        label=r"$d_{\max}$",
    )
    _add_switch_markers(ax, result)

    status_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.9, linestyle="-", label="established edge distance"),
        Line2D(
            [0],
            [0],
            color="0.25",
            linewidth=1.9,
            linestyle="--",
            label="prospective edge distance",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linewidth=1.45,
            linestyle="-.",
            label=r"relaxed upper bound $d_{\max}+\rho_k$",
        ),
    ]
    ax.set(
        xlabel="time [s]",
        ylabel="relative distance [m]",
        title="Inter-robot distances on established and prospective edges",
    )
    ax.grid(True, alpha=0.3)

    # The edge-specific legend makes the association between each distance and
    # its relaxed boundary explicit.  Matching colors identify the same edge;
    # line style identifies distance/status versus relaxed upper bound.
    first_legend = ax.legend(
        handles=[*edge_handles, *relaxed_bound_handles],
        title="Edge-specific signals",
        ncol=min(4, max(1, len(edge_handles))),
        loc="upper right",
        fontsize=8.3,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=[*status_handles, lower, upper],
        loc="lower right",
        fontsize=8.3,
    )

    path = output_dir / "edge_distances.png"
    _save_static_figure(fig, path)
    return path


def save_plots(
    result: SimulationResult,
    scenario: Scenario,
    output_dir: Path,
    virtual_speed_limit: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [_trajectory_plot(result, scenario, output_dir)]
    n = scenario.num_robots

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i in range(n):
        speed = np.linalg.norm(result.virtual_velocities[:, i], axis=1)
        ax.plot(
            result.time,
            np.where(result.controlled[:, i], speed, np.nan),
            label=f"robot {i + 1}",
        )
    ax.axhline(virtual_speed_limit, linestyle="--", label="virtual-speed limit")
    _add_switch_markers(ax, result)
    ax.set(xlabel="time [s]", ylabel="virtual speed [m/s]", title="Virtual velocities")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    path = output_dir / "virtual_speeds.png"
    _save_static_figure(fig, path)
    paths.append(path)

    paths.append(_edge_distance_plot(result, scenario, output_dir))

    # Relaxation states and relaxed-boundary margins.
    all_prospective = sorted(
        {edge for history in result.prospective_history for edge in history}
    )
    if all_prospective:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for edge in all_prospective:
            rho = np.array(
                [history.get(edge, np.nan) for history in result.prospective_history]
            )
            ax.plot(result.time, rho, label=rf"$\rho_{{{edge[0] + 1},{edge[1] + 1}}}$")
        ax.axhline(
            0.0,
            linestyle="--",
            label=r"activation at $\rho=0$",
        )
        _add_switch_markers(ax, result)
        ax.set(
            xlabel="time [s]",
            ylabel="relaxation [m]",
            title="Prospective-edge relaxation states",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        path = output_dir / "relaxation_states.png"
        _save_static_figure(fig, path)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(9, 4.6))
        for edge in all_prospective:
            margin = np.array(
                [
                    history.get(edge, np.nan)
                    for history in result.prospective_margin_history
                ]
            )
            ax.plot(result.time, margin, label=rf"$d_{{\max}}+\rho_{{{edge[0] + 1},{edge[1] + 1}}}-d_{{{edge[0] + 1},{edge[1] + 1}}}$")
        ax.axhline(
            scenario.reconfiguration.edge_activation_margin,
            linestyle="--",
            label=r"protected margin $\varepsilon_a$",
        )
        ax.axhline(0.0, linestyle=":", label="BLF boundary")
        _add_switch_markers(ax, result)
        ax.set(
            xlabel="time [s]",
            ylabel="margin [m]",
            title="Prospective-edge relaxed margins",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        path = output_dir / "relaxation_margins.png"
        _save_static_figure(fig, path)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(result.time, result.position_edge_error_norm, label="position edge-error norm")
    ax.plot(result.time, result.velocity_edge_error_norm, label="velocity edge-error norm")
    _add_switch_markers(ax, result)
    ax.set(
        xlabel="time [s]",
        ylabel="error norm",
        title="Established-edge synchronization errors",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "formation_errors.png"
    _save_static_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(result.time, result.lyapunov)
    _add_switch_markers(ax, result)
    ax.set(
        xlabel="time [s]",
        ylabel="V",
        title="Composite BLF candidate used by the controller",
    )
    ax.grid(True, alpha=0.3)
    path = output_dir / "lyapunov.png"
    _save_static_figure(fig, path)
    paths.append(path)

    if result.dynamics_model in ("quadrotor", "quadrotor_rotor"):
        assert result.quadrotor_config is not None
        assert result.thrusts is not None
        assert result.torques is not None
        assert result.attitude_errors is not None

        # Separate the coordination-to-flight interface from the actual
        # quadrotor realization.  This distinction is essential when debugging
        # physical command conditioning.
        fig, ax = plt.subplots(figsize=(9, 4.6))
        interface_error = result.controls - result.flight_acceleration_commands
        for i in range(n):
            tracking = np.linalg.norm(interface_error[:, i], axis=1)
            ax.plot(
                result.time,
                np.where(result.controlled[:, i], tracking, np.nan),
                label=f"robot {i + 1}",
            )
        _add_switch_markers(ax, result)
        ax.set(
            xlabel="time [s]",
            ylabel=r"$\|u_i-u_i^{cmd}\|$ [m/s$^2$]",
            title="Formation-command conditioning error",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        path = output_dir / "formation_command_conditioning_error.png"
        _save_static_figure(fig, path)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i in range(n):
            tracking = np.linalg.norm(
                result.flight_acceleration_tracking_error[:, i], axis=1
            )
            ax.plot(
                result.time,
                np.where(result.flight_controlled[:, i], tracking, np.nan),
                label=f"robot {i + 1}",
            )
        _add_switch_markers(ax, result)
        ax.set(
            xlabel="time [s]",
            ylabel=r"$\|u_i^{cmd}-a_i\|$ [m/s$^2$]",
            title="Quadrotor acceleration realization error",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        path = output_dir / "acceleration_realization_error.png"
        _save_static_figure(fig, path)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i in range(n):
            ax.plot(
                result.time,
                np.where(result.flight_controlled[:, i], result.thrusts[:, i], np.nan),
                label=f"robot {i + 1}",
            )
        cfg = result.quadrotor_config
        ax.axhline(cfg.mass * cfg.gravity, linestyle="--", color="0.35", label="hover thrust")
        ax.axhline(cfg.maximum_total_thrust, linestyle=":", color="0.25", label="collective limit")
        _add_switch_markers(ax, result)
        ax.set(xlabel="time [s]", ylabel="actual collective thrust [N]", title="Quadrotor collective thrust")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        path = output_dir / "quadrotor_thrusts.png"
        _save_static_figure(fig, path)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i in range(n):
            error_deg = np.rad2deg(result.attitude_errors[:, i])
            ax.plot(
                result.time,
                np.where(result.flight_controlled[:, i], error_deg, np.nan),
                label=f"robot {i + 1}",
            )
        _add_switch_markers(ax, result)
        ax.set(xlabel="time [s]", ylabel="attitude error [deg]", title="Low-level attitude tracking")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        path = output_dir / "quadrotor_attitude_error.png"
        _save_static_figure(fig, path)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i in range(n):
            torque_norm = np.linalg.norm(result.torques[:, i], axis=1)
            ax.plot(
                result.time,
                np.where(result.flight_controlled[:, i], torque_norm, np.nan),
                label=f"robot {i + 1}",
            )
        ax.axhline(
            cfg.maximum_torque,
            linestyle=":",
            color="0.25",
            label="torque-norm limit",
        )
        _add_switch_markers(ax, result)
        ax.set(xlabel="time [s]", ylabel="torque norm [N m]", title="Quadrotor body torques")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        path = output_dir / "quadrotor_torques.png"
        _save_static_figure(fig, path)
        paths.append(path)

    if result.dynamics_model == "quadrotor_rotor":
        assert result.rotor_thrusts is not None
        assert result.rotor_thrust_commands is not None
        assert result.rotor_allocation_saturated is not None
        assert result.quadrotor_config is not None

        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i in range(n):
            motor_lag = np.linalg.norm(
                result.rotor_thrust_commands[:, i] - result.rotor_thrusts[:, i],
                axis=1,
            )
            ax.plot(
                result.time,
                np.where(result.flight_controlled[:, i], motor_lag, np.nan),
                label=f"robot {i + 1}",
            )
        _add_switch_markers(ax, result)
        ax.set(
            xlabel="time [s]",
            ylabel=r"$\|T_i^{cmd}-T_i\|_2$ [N]",
            title="First-order rotor-thrust tracking",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        path = output_dir / "rotor_thrust_tracking.png"
        _save_static_figure(fig, path)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(9, 4.6))
        cfg = result.quadrotor_config
        for i in range(n):
            maximum_rotor = np.max(result.rotor_thrusts[:, i], axis=1)
            ax.plot(
                result.time,
                np.where(result.flight_controlled[:, i], maximum_rotor, np.nan),
                label=f"robot {i + 1}",
            )
        ax.axhline(
            cfg.maximum_rotor_thrust,
            linestyle=":",
            color="0.25",
            label="per-rotor limit",
        )
        _add_switch_markers(ax, result)
        ax.set(
            xlabel="time [s]",
            ylabel="maximum rotor thrust [N]",
            title="Individual rotor loading",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        path = output_dir / "rotor_loading.png"
        _save_static_figure(fig, path)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    active_count = result.active.sum(axis=1)
    edge_count = np.array([len(edges) for edges in result.edge_history])
    prospective_count = np.array([len(edges) for edges in result.prospective_history])
    staging_count = result.staging.sum(axis=1)
    returning_count = result.returning.sum(axis=1)
    parked_count = result.parked.sum(axis=1)
    ax.step(result.time, active_count, where="post", label="team robots")
    ax.step(result.time, staging_count, where="post", label="staging robots")
    ax.step(result.time, returning_count, where="post", label="returning robots")
    ax.step(result.time, parked_count, where="post", label="charging robots")
    ax.step(result.time, edge_count, where="post", label="established edges")
    ax.step(result.time, prospective_count, where="post", label="prospective edges")
    _add_switch_markers(ax, result)
    ax.set(xlabel="time [s]", ylabel="count", title="Open-team reconfiguration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "open_system_counts_diagnostic.png"
    _save_static_figure(fig, path)
    paths.append(path)

    # Dedicated manuscript figures use the paper's notation, actual LaTeX
    # rendering when available, MATLAB's ColorOrder, larger final-size fonts,
    # and thicker lines.  They are intentionally kept separate from the richer
    # diagnostic plots above.
    from .paper_plotting import save_paper_plots

    paths.extend(save_paper_plots(result, scenario, output_dir))

    # Every static plot is saved both as PNG for quick inspection and as a
    # vector PDF suitable for direct inclusion in the paper.
    expanded_paths: list[Path] = []
    for png_path in paths:
        expanded_paths.extend((png_path, png_path.with_suffix(".pdf")))
    return expanded_paths


def save_animation(
    result: SimulationResult,
    scenario: Scenario,
    output_path: Path,
    fps: int = 15,
    stride: int = 10,
    *,
    tail_seconds: float = 5.0,
    rotate_3d_camera: bool = True,
) -> Path:
    """Save an animation with dashed green established and dashed orange prospective edges.

    The 3-D rendering additionally shows short trajectory tails and a
    translationally aligned representative of the current desired formation.
    Since the controller regulates only relative displacements, this gauge fixing
    makes convergence to the desired formation visually meaningful.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = np.arange(0, len(result.time), max(1, int(stride)))
    all_edges = _edge_union(result)
    cmap = plt.get_cmap("tab10")
    tail_samples = max(1, int(round(tail_seconds / scenario.dt)))

    if scenario.dimension == 2:
        fig, ax = plt.subplots(figsize=(8.0, 6.5))
        points = result.positions[result.controlled]
        mins = np.min(points, axis=0) - 0.6
        maxs = np.max(points, axis=0) + 0.6
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        robot_artists = [
            ax.plot([], [], "o", color=cmap(i % 10), markersize=8)[0]
            for i in range(scenario.num_robots)
        ]
        edge_artists = {edge: ax.plot([], [], linewidth=1.8)[0] for edge in all_edges}
        time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

        def update(frame: int):
            k = int(frame_ids[frame])
            for i, artist in enumerate(robot_artists):
                visible = bool(result.controlled[k, i])
                artist.set_visible(visible)
                if visible:
                    artist.set_data(
                        [result.positions[k, i, 0]],
                        [result.positions[k, i, 1]],
                    )
                    artist.set_markerfacecolor(
                        cmap(i % 10) if result.active[k, i] else "none"
                    )
            established = set(result.edge_history[k])
            prospective = set(result.prospective_history[k])
            for edge, artist in edge_artists.items():
                if edge not in established and edge not in prospective:
                    artist.set_visible(False)
                    continue
                i, j = edge
                pi, pj = result.positions[k, i], result.positions[k, j]
                artist.set_data([pi[0], pj[0]], [pi[1], pj[1]])
                artist.set_linestyle("--")
                artist.set_color("tab:green" if edge in established else "tab:orange")
                artist.set_visible(True)
            time_text.set_text(f"t={result.time[k]:.1f} s")
            return [*robot_artists, *edge_artists.values(), time_text]

    else:
        fig = plt.figure(figsize=(9.4, 7.8))
        ax = fig.add_subplot(111, projection="3d")
        physical_visible = result.flight_controlled | result.parked
        _set_3d_equal_box(ax, result.positions[physical_visible], padding=0.65)
        _draw_charging_area(ax, scenario)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.grid(True, alpha=0.25)
        ax.view_init(elev=24.0, azim=-58.0)

        use_quadrotor_mesh = (
            result.attitudes is not None and result.quadrotor_config is not None
        )
        if use_quadrotor_mesh:
            drone_artists = [
                _create_quadrotor_artists(
                    ax,
                    cmap(i % 10),
                    result.quadrotor_config,
                )
                for i in range(scenario.num_robots)
            ]
            robot_artists = []
        else:
            drone_artists = []
            robot_artists = [
                ax.plot([], [], [], "o", color=cmap(i % 10), markersize=8)[0]
                for i in range(scenario.num_robots)
            ]
        trail_artists = [
            ax.plot([], [], [], color=cmap(i % 10), linewidth=1.2, alpha=0.45)[0]
            for i in range(scenario.num_robots)
        ]
        desired_artists = [
            ax.plot(
                [],
                [],
                [],
                marker="+",
                linestyle="None",
                color=cmap(i % 10),
                markersize=8,
                alpha=0.35,
            )[0]
            for i in range(scenario.num_robots)
        ]
        edge_artists = {
            edge: ax.plot([], [], [], linewidth=2.0)[0]
            for edge in all_edges
        }
        status_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, va="top")

        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="tab:green",
                    linewidth=2.0,
                    linestyle="--",
                    label="established edge",
                ),
                Line2D(
                    [0],
                    [0],
                    color="tab:orange",
                    linewidth=2.0,
                    linestyle="--",
                    label="prospective edge",
                ),
                Line2D([0], [0], marker="o", color="0.35", linestyle="None", label="active robot"),
                Line2D([0], [0], marker="o", markerfacecolor="none", color="0.35", linestyle="None", label="staging/returning robot"),
                Line2D([0], [0], marker="s", markerfacecolor="none", color="0.55", linestyle="None", label="charging pad / parked robot"),
                Line2D(
                    [0],
                    [0],
                    marker="+",
                    color="0.35",
                    linestyle="None",
                    label="desired position (translation-aligned)",
                ),
            ],
            loc="upper right",
            fontsize=8.5,
        )

        def update(frame: int):
            k = int(frame_ids[frame])
            tail_start = max(0, k - tail_samples)

            aligned_desired = _aligned_desired_positions(result, k)

            for i, (trail, desired) in enumerate(
                zip(trail_artists, desired_artists)
            ):
                visible = bool(result.flight_controlled[k, i] or result.parked[k, i])
                trail_visible = bool(result.flight_controlled[k, i])
                trail.set_visible(trail_visible)
                desired.set_visible(bool(result.controlled[k, i]))

                if use_quadrotor_mesh:
                    assert result.attitudes is not None
                    assert result.quadrotor_config is not None
                    if result.active[k, i]:
                        opacity = 0.98
                    elif result.returning[k, i]:
                        opacity = 0.80
                    elif result.staging[k, i]:
                        opacity = 0.72
                    else:
                        opacity = 0.38
                    _set_quadrotor_artists(
                        drone_artists[i],
                        result.positions[k, i],
                        result.attitudes[k, i],
                        result.quadrotor_config,
                        visible=visible,
                        active=bool(result.active[k, i]),
                        opacity=opacity,
                    )
                else:
                    robot = robot_artists[i]
                    robot.set_visible(visible)
                    if visible:
                        p = result.positions[k, i]
                        robot.set_data_3d([p[0]], [p[1]], [p[2]])
                        robot.set_markerfacecolor(
                            cmap(i % 10) if result.active[k, i] else "none"
                        )
                        robot.set_markeredgewidth(
                            1.5 if not result.active[k, i] else 1.0
                        )

                if not visible:
                    continue

                trail_mask = result.flight_controlled[tail_start : k + 1, i]
                trail_points = result.positions[tail_start : k + 1, i][trail_mask]
                if len(trail_points):
                    trail.set_data_3d(
                        trail_points[:, 0],
                        trail_points[:, 1],
                        trail_points[:, 2],
                    )

                if result.controlled[k, i]:
                    p_des = aligned_desired[i]
                    desired.set_data_3d([p_des[0]], [p_des[1]], [p_des[2]])

            established = set(result.edge_history[k])
            prospective = set(result.prospective_history[k])
            for edge, artist in edge_artists.items():
                if edge not in established and edge not in prospective:
                    artist.set_visible(False)
                    continue
                i, j = edge
                pi, pj = result.positions[k, i], result.positions[k, j]
                artist.set_data_3d(
                    [pi[0], pj[0]],
                    [pi[1], pj[1]],
                    [pi[2], pj[2]],
                )
                artist.set_linestyle("--")
                artist.set_color("tab:green" if edge in established else "tab:orange")
                artist.set_alpha(0.9)
                artist.set_visible(True)

            if rotate_3d_camera:
                phase = frame / max(1, len(frame_ids) - 1)
                ax.view_init(
                    elev=24.0 + 3.0 * np.sin(2.0 * np.pi * phase),
                    azim=-62.0 + 28.0 * phase,
                )

            status_text.set_text(
                f"t = {result.time[k]:.1f} s\n"
                f"team: {int(result.active[k].sum())} robots   "
                f"edges: {len(established)}   prospective: {len(prospective)}\n"
                f"plant: {result.dynamics_model}"
            )
            drone_flat = [artist for group in drone_artists for artist in group]
            return [
                *robot_artists,
                *drone_flat,
                *trail_artists,
                *desired_artists,
                *edge_artists.values(),
                status_text,
            ]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000 / fps,
        blit=False,
    )
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not writers.is_available("ffmpeg"):
            plt.close(fig)
            raise RuntimeError("ffmpeg is unavailable; cannot write MP4 animation.")
        writer = FFMpegWriter(
            fps=fps,
            bitrate=2400,
            metadata={"title": "Open multi-robot system simulation"},
        )
    else:
        plt.close(fig)
        raise ValueError("Animation output must use .gif or .mp4.")
    animation.save(output_path, writer=writer)
    plt.close(fig)
    return output_path
