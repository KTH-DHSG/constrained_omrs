from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .graph import Edge
from .paper_plotting import (
    DOUBLE_COLUMN_WIDTH_IN,
    MATLAB_COLORS,
    _edge_distance,
    _edge_status_masks,
    _edge_union,
    _event_code,
    _extended_matlab_colors,
    _finish_time_axis,
    paper_style,
)
from .scenario import Scenario
from .simulation import SimulationResult


@dataclass(frozen=True)
class PaperAnimationConfig:
    """Rendering options for manuscript-style plot animations.

    ``playback_speed`` maps simulation seconds to video seconds.  A value of
    1.0 preserves the Gazebo / simulation time base exactly; 2.0 produces a
    video that is twice as fast.  Keeping this at 1.0 is convenient when the
    plot clips are synchronized with the Gazebo recording in a video editor.
    """

    fps: int = 30
    dpi: int = 180
    playback_speed: float = 1.0
    bitrate_kbps: int = 6500
    start_time: float | None = None
    end_time: float | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if self.playback_speed <= 0.0:
            raise ValueError("playback_speed must be positive.")
        if self.bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be positive.")


def _time_window(result: SimulationResult, config: PaperAnimationConfig) -> tuple[float, float]:
    t0 = float(result.time[0]) if config.start_time is None else float(config.start_time)
    tf = float(result.time[-1]) if config.end_time is None else float(config.end_time)
    t0 = max(t0, float(result.time[0]))
    tf = min(tf, float(result.time[-1]))
    if tf <= t0:
        raise ValueError(f"Invalid animation time window [{t0}, {tf}].")
    return t0, tf


def _frame_times(
    result: SimulationResult,
    config: PaperAnimationConfig,
) -> np.ndarray:
    t0, tf = _time_window(result, config)
    duration_video = (tf - t0) / config.playback_speed
    frame_count = max(2, int(np.ceil(duration_video * config.fps)) + 1)
    return np.linspace(t0, tf, frame_count)


def _sample_index(time: np.ndarray, current_time: float) -> int:
    return int(np.clip(np.searchsorted(time, current_time, side="right") - 1, 0, len(time) - 1))


def _fixed_ylim(ax, values: np.ndarray, *, lower_zero: bool = False, padding: float = 0.07) -> None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        ax.set_ylim(0.0, 1.0)
        return
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lower_zero:
        lo = min(0.0, lo)
    span = max(hi - lo, 1e-6)
    lo_out = lo - padding * span
    hi_out = hi + padding * span
    if lower_zero and lo >= -1e-12:
        lo_out = 0.0
    ax.set_ylim(lo_out, hi_out)


def _make_event_artists(axes, result: SimulationResult):
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    axes = list(np.ravel(axes))
    lines: list[tuple[float, list[Line2D]]] = []
    labels: list[tuple[float, object]] = []
    for event_index, event in enumerate(result.switch_events):
        per_axis: list[Line2D] = []
        for ax in axes:
            artist = ax.axvline(
                float(event.time),
                color="0.48",
                linewidth=0.95,
                linestyle=(0, (2.0, 2.4)),
                alpha=0.75,
                zorder=0,
            )
            artist.set_visible(False)
            per_axis.append(artist)
        lines.append((float(event.time), per_axis))

        y = 1.015 + 0.055 * (event_index % 2)
        text = axes[0].text(
            float(event.time),
            y,
            _event_code(event),
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="0.28",
            clip_on=False,
        )
        text.set_visible(False)
        labels.append((float(event.time), text))
    return lines, labels


def _set_event_visibility(event_lines, event_labels, current_time: float) -> None:
    for event_time, artists in event_lines:
        visible = event_time <= current_time + 1e-12
        for artist in artists:
            artist.set_visible(visible)
    for event_time, artist in event_labels:
        artist.set_visible(event_time <= current_time + 1e-12)


def _save_mp4(fig, update, frame_times: np.ndarray, path: Path, config: PaperAnimationConfig) -> Path:
    if shutil.which("ffmpeg") is None or not animation.writers.is_available("ffmpeg"):
        plt.close(fig)
        raise RuntimeError(
            "ffmpeg is required for paper plot animations. Install it with "
            "`sudo apt install ffmpeg`."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=config.fps,
        bitrate=config.bitrate_kbps,
        codec="libx264",
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": path.stem, "artist": "constrained_omrs"},
    )
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frame_times,
        interval=1000.0 / config.fps,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )
    movie.save(path, writer=writer, dpi=config.dpi)
    plt.close(fig)
    return path


def animate_formation_performance(
    result: SimulationResult,
    output_path: Path,
    config: PaperAnimationConfig = PaperAnimationConfig(),
) -> Path:
    """Animate the three-panel formation-performance figure from the paper."""
    t0, tf = _time_window(result, config)
    frames = _frame_times(result, config)

    with paper_style():
        fig, axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.75),
            gridspec_kw={"hspace": 0.10},
        )
        blue, orange, yellow = MATLAB_COLORS[:3]
        line_pos, = axes[0].plot([], [], color=blue, linewidth=2.35)
        line_vel, = axes[1].plot([], [], color=orange, linewidth=2.35)
        line_v, = axes[2].plot([], [], color=yellow, linewidth=2.35)

        axes[0].set_ylabel(r"$\lvert e_{x,\phi}\rvert\;[\mathrm{m}]$")
        axes[1].set_ylabel(
            r"$\lvert [E_\phi^\top\!\otimes I_3]v_\phi\rvert$"
            "\n" r"$[\mathrm{m}\,\mathrm{s}^{-1}]$"
        )
        axes[2].set_ylabel(r"$V_\phi$")

        _fixed_ylim(axes[0], result.position_edge_error_norm, lower_zero=True)
        _fixed_ylim(axes[1], result.velocity_edge_error_norm, lower_zero=True)
        _fixed_ylim(axes[2], result.lyapunov, lower_zero=True)

        event_lines, event_labels = _make_event_artists(axes, result)
        for index, ax in enumerate(axes):
            _finish_time_axis(ax, result, xlabel=index == len(axes) - 1)
            ax.set_xlim(t0, tf)
            if index < len(axes) - 1:
                ax.tick_params(labelbottom=False)
        fig.subplots_adjust(left=0.12, right=0.99, bottom=0.10, top=0.925)

        def update(current_time: float):
            k = _sample_index(result.time, float(current_time))
            sl = slice(0, k + 1)
            line_pos.set_data(result.time[sl], result.position_edge_error_norm[sl])
            line_vel.set_data(result.time[sl], result.velocity_edge_error_norm[sl])
            line_v.set_data(result.time[sl], result.lyapunov[sl])
            _set_event_visibility(event_lines, event_labels, float(current_time))
            return (line_pos, line_vel, line_v)

        return _save_mp4(fig, update, frames, output_path, config)


def animate_open_system_counts(
    result: SimulationResult,
    output_path: Path,
    config: PaperAnimationConfig = PaperAnimationConfig(),
) -> Path:
    """Animate the two-panel robot / edge count figure from the paper."""
    t0, tf = _time_window(result, config)
    frames = _frame_times(result, config)

    active_count = result.active.sum(axis=1)
    staging_count = result.staging.sum(axis=1)
    returning_count = result.returning.sum(axis=1)
    parked_count = result.parked.sum(axis=1)
    established_count = np.asarray([len(edges) for edges in result.edge_history], dtype=int)
    prospective_count = np.asarray([len(edges) for edges in result.prospective_history], dtype=int)

    with paper_style():
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.65),
            gridspec_kw={"height_ratios": [1.08, 0.92], "hspace": 0.10},
        )
        blue, orange, yellow, purple, green = MATLAB_COLORS[:5]
        l_active, = axes[0].step([], [], where="post", color=blue, linewidth=2.50,
                                 label=r"OMRS robots $|\mathcal V_\phi|$")
        l_stage, = axes[0].step([], [], where="post", color=yellow, linewidth=2.25,
                                label="joining robots")
        l_return, = axes[0].step([], [], where="post", color=green, linewidth=2.25,
                                 label="returning robots")
        l_parked, = axes[0].step([], [], where="post", color=purple, linewidth=2.25,
                                 label="charging robots")
        axes[0].set_ylabel("robots")
        max_robots = int(max(np.max(active_count), np.max(staging_count),
                             np.max(returning_count), np.max(parked_count)))
        axes[0].set_ylim(-0.25, max_robots + 0.55)
        axes[0].set_yticks(np.arange(0, max_robots + 1, 1))
        axes[0].legend(ncol=2, loc="upper left", columnspacing=1.25,
                       handlelength=2.6, handletextpad=0.55)

        l_edges, = axes[1].step([], [], where="post", color=green, linewidth=2.50,
                                label=r"established $|\mathcal E_\phi|$")
        l_prospective, = axes[1].step([], [], where="post", color=orange, linewidth=2.50,
                                      label=r"prospective $|\mathcal E_\phi^{\mathrm p}|$")
        axes[1].set_ylabel("edges")
        max_edges = int(max(np.max(established_count), np.max(prospective_count)))
        axes[1].set_ylim(-0.25, max_edges + 0.55)
        axes[1].set_yticks(np.arange(0, max_edges + 1, 1 if max_edges <= 6 else 2))
        axes[1].legend(ncol=2, loc="upper left", columnspacing=1.25,
                       handlelength=2.6, handletextpad=0.55)

        event_lines, event_labels = _make_event_artists(axes, result)
        for index, ax in enumerate(axes):
            _finish_time_axis(ax, result, xlabel=index == 1)
            ax.set_xlim(t0, tf)
            if index == 0:
                ax.tick_params(labelbottom=False)
        fig.subplots_adjust(left=0.105, right=0.99, bottom=0.115, top=0.90)

        def update(current_time: float):
            k = _sample_index(result.time, float(current_time))
            sl = slice(0, k + 1)
            t = result.time[sl]
            l_active.set_data(t, active_count[sl])
            l_stage.set_data(t, staging_count[sl])
            l_return.set_data(t, returning_count[sl])
            l_parked.set_data(t, parked_count[sl])
            l_edges.set_data(t, established_count[sl])
            l_prospective.set_data(t, prospective_count[sl])
            _set_event_visibility(event_lines, event_labels, float(current_time))
            return (l_active, l_stage, l_return, l_parked, l_edges, l_prospective)

        return _save_mp4(fig, update, frames, output_path, config)


def animate_edge_distances_detailed(
    result: SimulationResult,
    scenario: Scenario,
    output_path: Path,
    config: PaperAnimationConfig = PaperAnimationConfig(),
) -> Path:
    """Animate the detailed edge-distance / relaxed-bound figure from the paper."""
    t0, tf = _time_window(result, config)
    frames = _frame_times(result, config)
    edges = _edge_union(result)
    colors = _extended_matlab_colors(len(edges))

    edge_data = {}
    max_relaxed = float(scenario.sensing_distance)
    for edge in edges:
        d = _edge_distance(result, edge)
        established, prospective = _edge_status_masks(result, edge)
        rho = np.asarray(
            [history.get(edge, np.nan) for history in result.prospective_history],
            dtype=float,
        )
        relaxed = scenario.sensing_distance + rho
        if np.any(np.isfinite(relaxed)):
            max_relaxed = max(max_relaxed, float(np.nanmax(relaxed)))
        edge_data[edge] = (d, established, prospective, relaxed)

    with paper_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.35))
        edge_handles: list[Line2D] = []
        artists = {}

        for color, edge in zip(colors, edges):
            established_line, = ax.plot([], [], color=color, linewidth=2.30,
                                        linestyle="-", solid_capstyle="round", zorder=3)
            prospective_line, = ax.plot([], [], color=color, linewidth=2.30,
                                        linestyle=(0, (5.5, 2.5)), solid_capstyle="round", zorder=3)
            relaxed_line, = ax.plot([], [], color=color, linewidth=1.75,
                                    linestyle=(0, (4.5, 1.8, 1.2, 1.8)), alpha=0.78, zorder=2)
            artists[edge] = (established_line, prospective_line, relaxed_line)
            edge_handles.append(Line2D([0], [0], color=color, linewidth=2.5,
                                       label=rf"$({edge[0]+1},{edge[1]+1})$"))

        ax.axhline(scenario.collision_distance, color="0.08", linewidth=2.05,
                   linestyle=(0, (7, 3)), zorder=1)
        ax.axhline(scenario.sensing_distance, color="0.30", linewidth=2.05,
                   linestyle=(0, (2.2, 2.0)), zorder=1)

        x_label = tf - 0.012 * (tf - t0)
        ax.text(x_label, scenario.collision_distance + 0.025, r"$d_{\min}$",
                ha="right", va="bottom", fontsize=10.5, color="0.08")
        ax.text(x_label, scenario.sensing_distance + 0.025, r"$d_{\max}$",
                ha="right", va="bottom", fontsize=10.5, color="0.25")

        event_lines, event_labels = _make_event_artists(ax, result)
        _finish_time_axis(ax, result, xlabel=True)
        ax.set_xlim(t0, tf)
        ax.set_ylabel(r"$d_k\;[\mathrm{m}]$")
        upper = max(max_relaxed + 0.10, scenario.sensing_distance + 0.25)
        ax.set_ylim(max(0.0, scenario.collision_distance - 0.16), upper)

        fig.legend(handles=edge_handles, title=r"edge $\varepsilon_k=(i,j)$",
                   loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=6,
                   columnspacing=1.25, handlelength=2.3, handletextpad=0.55,
                   fontsize=9.6, title_fontsize=10.2, frameon=False)
        style_handles = [
            Line2D([0], [0], color="0.25", linewidth=2.3, linestyle="-", label="established"),
            Line2D([0], [0], color="0.25", linewidth=2.3,
                   linestyle=(0, (5.5, 2.5)), label="prospective"),
            Line2D([0], [0], color="0.25", linewidth=1.75,
                   linestyle=(0, (4.5, 1.8, 1.2, 1.8)), label=r"$d_{\max}+\rho_k$"),
        ]
        ax.legend(handles=style_handles, loc="upper left", ncol=3,
                  handlelength=2.7, columnspacing=1.3, fontsize=9.3)
        fig.subplots_adjust(left=0.105, right=0.99, bottom=0.285, top=0.90)

        def update(current_time: float):
            k = _sample_index(result.time, float(current_time))
            sl = slice(0, k + 1)
            t = result.time[sl]
            changed = []
            for edge in edges:
                d, established, prospective, relaxed = edge_data[edge]
                established_line, prospective_line, relaxed_line = artists[edge]
                established_line.set_data(t, np.where(established[sl], d[sl], np.nan))
                prospective_line.set_data(t, np.where(prospective[sl], d[sl], np.nan))
                relaxed_visible = np.where(prospective[sl], relaxed[sl], np.nan)
                relaxed_line.set_data(t, relaxed_visible)
                changed.extend((established_line, prospective_line, relaxed_line))
            _set_event_visibility(event_lines, event_labels, float(current_time))
            return tuple(changed)

        return _save_mp4(fig, update, frames, output_path, config)


def save_paper_animations(
    result: SimulationResult,
    scenario: Scenario,
    output_dir: Path,
    *,
    config: PaperAnimationConfig = PaperAnimationConfig(),
    selected: tuple[str, ...] = ("edge_distances", "formation_performance", "open_system_counts"),
) -> list[Path]:
    """Render the three video plots corresponding to the paper's results figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = {"edge_distances", "formation_performance", "open_system_counts"}
    unknown = set(selected) - valid
    if unknown:
        raise ValueError(f"Unknown paper animation(s): {sorted(unknown)}")

    paths: list[Path] = []
    if "edge_distances" in selected:
        paths.append(
            animate_edge_distances_detailed(
                result, scenario, output_dir / "paper_edge_distances_detailed.mp4", config
            )
        )
    if "formation_performance" in selected:
        paths.append(
            animate_formation_performance(
                result, output_dir / "paper_formation_performance.mp4", config
            )
        )
    if "open_system_counts" in selected:
        paths.append(
            animate_open_system_counts(
                result, output_dir / "open_system_counts.mp4", config
            )
        )
    return paths
