from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .graph import Edge
from .scenario import Scenario
from .simulation import SimulationResult


# MATLAB's long-standing default ColorOrder.  Keeping this centralized makes
# every paper figure visually consistent and prevents ad-hoc color choices.
MATLAB_COLORS = np.asarray(
    [
        [0.0000, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ],
    dtype=float,
)

# IEEE double-column width is approximately 7.0--7.2 in depending on the
# template.  The figures are authored at their final intended size, so fonts
# are not made tiny and then enlarged by LaTeX.
DOUBLE_COLUMN_WIDTH_IN = 7.10
SINGLE_COLUMN_WIDTH_IN = 3.45


@contextmanager
def paper_style():
    """Temporarily apply the manuscript-oriented Matplotlib style.

    When a TeX installation is available we let LaTeX typeset all labels, so
    the figures use the same Computer-Modern/IEEE visual language as the paper.
    On minimal machines the fallback uses Matplotlib's Computer Modern mathtext
    instead of failing post-processing entirely.
    """
    use_tex = shutil.which("latex") is not None
    rc = {
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.size": 11.0,
        "axes.labelsize": 12.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.15,
        "lines.markersize": 5.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "legend.frameon": False,
        "grid.color": "0.84",
        "grid.linewidth": 0.65,
        "grid.alpha": 0.55,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
        "axes.prop_cycle": mpl.cycler(color=MATLAB_COLORS.tolist()),
    }
    if use_tex:
        rc["text.latex.preamble"] = (
            r"\usepackage{amsmath,amssymb,amsfonts,mathtools}"
        )
    with mpl.rc_context(rc):
        yield


def _extended_matlab_colors(count: int) -> list[tuple[float, float, float]]:
    """Return enough distinguishable MATLAB-derived colors for many edges."""
    colors = [tuple(c) for c in MATLAB_COLORS]
    if count <= len(colors):
        return colors[:count]

    # The detailed edge plot can contain more than seven historical edges.
    # Continue with darker versions of the same MATLAB hues rather than
    # introducing a visually unrelated palette.
    for factor in (0.72, 0.52):
        for base in MATLAB_COLORS:
            colors.append(tuple(np.clip(factor * base, 0.0, 1.0)))
            if len(colors) >= count:
                return colors
    return colors[:count]


def _save(fig, output_dir: Path, stem: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return png


def _edge_union(result: SimulationResult) -> list[Edge]:
    edges = {edge for history in result.edge_history for edge in history}
    edges.update(edge for history in result.prospective_history for edge in history)
    return sorted(edges)


def _edge_distance(result: SimulationResult, edge: Edge) -> np.ndarray:
    i, j = edge
    return np.linalg.norm(result.positions[:, i] - result.positions[:, j], axis=1)


def _edge_status_masks(result: SimulationResult, edge: Edge) -> tuple[np.ndarray, np.ndarray]:
    established = np.asarray([edge in h for h in result.edge_history], dtype=bool)
    prospective = np.asarray([edge in h for h in result.prospective_history], dtype=bool)
    return established, prospective


def _event_code(event) -> str:
    robot = int(event.robot) + 1 if event.robot is not None else None
    if event.kind == "robot_join":
        return rf"$J_{{{robot}}}$"
    if event.kind == "bridge_activation":
        return rf"$B_{{{robot}}}$"
    if event.kind == "robot_leave":
        return rf"$L_{{{robot}}}$"
    return ""


def _add_event_lines(axes, result: SimulationResult, *, labels: bool = True) -> None:
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    axes = list(np.ravel(axes))
    if not axes:
        return

    for event_index, event in enumerate(result.switch_events):
        for ax in axes:
            ax.axvline(
                float(event.time),
                color="0.48",
                linewidth=0.95,
                linestyle=(0, (2.0, 2.4)),
                alpha=0.75,
                zorder=0,
            )
        if labels:
            y = 1.015 + 0.055 * (event_index % 2)
            axes[0].text(
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


def _finish_time_axis(ax, result: SimulationResult, *, xlabel: bool = True) -> None:
    ax.set_xlim(float(result.time[0]), float(result.time[-1]))
    if xlabel:
        ax.set_xlabel(r"$t\;[\mathrm{s}]$")
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def paper_formation_performance(
    result: SimulationResult,
    output_dir: Path,
) -> Path:
    """Position/velocity synchronization errors and composite BLF."""
    with paper_style():
        fig, axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.75),
            gridspec_kw={"hspace": 0.10},
        )

        blue, orange, yellow = MATLAB_COLORS[:3]
        axes[0].plot(result.time, result.position_edge_error_norm, color=blue, linewidth=2.35)
        axes[0].set_ylabel(r"$\lvert e_{x,\phi}\rvert\;[\mathrm{m}]$")

        axes[1].plot(result.time, result.velocity_edge_error_norm, color=orange, linewidth=2.35)
        axes[1].set_ylabel(
            r"$\lvert [E_\phi^\top\!\otimes I_3]v_\phi\rvert$"
            "\n" r"$[\mathrm{m}\,\mathrm{s}^{-1}]$"
        )

        axes[2].plot(result.time, result.lyapunov, color=yellow, linewidth=2.35)
        axes[2].set_ylabel(r"$V_\phi$")

        _add_event_lines(axes, result, labels=True)
        for index, ax in enumerate(axes):
            _finish_time_axis(ax, result, xlabel=index == len(axes) - 1)
            if index < len(axes) - 1:
                ax.tick_params(labelbottom=False)
        fig.subplots_adjust(left=0.12, right=0.99, bottom=0.10, top=0.925)
        return _save(fig, output_dir, "paper_formation_performance")


def paper_constraint_summary(
    result: SimulationResult,
    scenario: Scenario,
    output_dir: Path,
) -> Path:
    """Compact constraint verification and prospective-edge relaxation plot."""
    with paper_style():
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.20),
            gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
        )
        blue, orange = MATLAB_COLORS[:2]

        axes[0].plot(
            result.time,
            result.min_pair_distance,
            color=blue,
            linewidth=2.35,
            label=r"$\min_{\substack{i,j\in\mathcal V_\phi\\i\neq j}}\lvert x_i-x_j\rvert$",
        )
        axes[0].plot(
            result.time,
            result.max_established_edge_distance,
            color=orange,
            linewidth=2.35,
            label=r"$\max_{\varepsilon_k\in\mathcal E_\phi}d_k$",
        )
        axes[0].axhline(
            scenario.collision_distance,
            color="0.12",
            linewidth=2.0,
            linestyle=(0, (6, 3)),
            label=r"$d_{\min}$",
        )
        axes[0].axhline(
            scenario.sensing_distance,
            color="0.38",
            linewidth=2.0,
            linestyle=(0, (2, 2)),
            label=r"$d_{\max}$",
        )
        axes[0].set_ylabel(r"distance $[\mathrm{m}]$")
        axes[0].legend(loc="best", ncol=2, columnspacing=1.25, handlelength=2.7)

        prospective_edges = sorted(
            {edge for history in result.prospective_history for edge in history}
        )
        colors = _extended_matlab_colors(len(prospective_edges))
        for color, edge in zip(colors, prospective_edges):
            rho = np.asarray(
                [history.get(edge, np.nan) for history in result.prospective_history],
                dtype=float,
            )
            axes[1].plot(
                result.time,
                rho,
                color=color,
                linewidth=2.2,
                label=rf"$\varepsilon_k=({edge[0]+1},{edge[1]+1})$",
            )
        axes[1].axhline(0.0, color="0.2", linewidth=1.5, linestyle=(0, (3, 2)))
        axes[1].set_ylabel(r"$\rho_k\;[\mathrm{m}]$")
        if prospective_edges:
            axes[1].legend(
                loc="upper right",
                ncol=min(4, max(1, len(prospective_edges))),
                columnspacing=1.0,
                handlelength=2.2,
                fontsize=8.9,
            )

        _add_event_lines(axes, result, labels=True)
        for index, ax in enumerate(axes):
            _finish_time_axis(ax, result, xlabel=index == 1)
            if index == 0:
                ax.tick_params(labelbottom=False)
        fig.subplots_adjust(left=0.115, right=0.99, bottom=0.105, top=0.92)
        return _save(fig, output_dir, "paper_constraint_summary")


def paper_edge_distances_detailed(
    result: SimulationResult,
    scenario: Scenario,
    output_dir: Path,
) -> Path:
    """Detailed all-edge constraint plot retaining the full diagnostic content.

    Solid segments are established interactions; dashed segments are prospective
    interactions.  During a prospective phase the corresponding relaxed upper
    bound d_max + rho_k is shown with a thin dash-dot curve in the same color.
    """
    edges = _edge_union(result)
    colors = _extended_matlab_colors(len(edges))

    with paper_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.35))
        edge_handles: list[Line2D] = []

        max_relaxed = float(scenario.sensing_distance)
        for color, edge in zip(colors, edges):
            d = _edge_distance(result, edge)
            established, prospective = _edge_status_masks(result, edge)

            ax.plot(
                result.time,
                np.where(established, d, np.nan),
                color=color,
                linewidth=2.30,
                linestyle="-",
                solid_capstyle="round",
                zorder=3,
            )
            ax.plot(
                result.time,
                np.where(prospective, d, np.nan),
                color=color,
                linewidth=2.30,
                linestyle=(0, (5.5, 2.5)),
                solid_capstyle="round",
                zorder=3,
            )

            rho = np.asarray(
                [history.get(edge, np.nan) for history in result.prospective_history],
                dtype=float,
            )
            finite = np.isfinite(rho)
            if np.any(finite):
                relaxed = scenario.sensing_distance + rho
                max_relaxed = max(max_relaxed, float(np.nanmax(relaxed)))
                ax.plot(
                    result.time,
                    relaxed,
                    color=color,
                    linewidth=1.75,
                    linestyle=(0, (4.5, 1.8, 1.2, 1.8)),
                    alpha=0.78,
                    zorder=2,
                )

            edge_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.5,
                    label=rf"$({edge[0]+1},{edge[1]+1})$",
                )
            )

        ax.axhline(
            scenario.collision_distance,
            color="0.08",
            linewidth=2.05,
            linestyle=(0, (7, 3)),
            zorder=1,
        )
        ax.axhline(
            scenario.sensing_distance,
            color="0.30",
            linewidth=2.05,
            linestyle=(0, (2.2, 2.0)),
            zorder=1,
        )

        # Direct labels avoid spending legend entries on the two global bounds.
        x_label = float(result.time[-1]) - 0.012 * (float(result.time[-1]) - float(result.time[0]))
        ax.text(
            x_label,
            scenario.collision_distance + 0.025,
            r"$d_{\min}$",
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="0.08",
        )
        ax.text(
            x_label,
            scenario.sensing_distance + 0.025,
            r"$d_{\max}$",
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="0.25",
        )

        _add_event_lines(ax, result, labels=True)
        _finish_time_axis(ax, result, xlabel=True)
        ax.set_ylabel(r"$d_k\;[\mathrm{m}]$")
        upper = max(max_relaxed + 0.10, scenario.sensing_distance + 0.25)
        ax.set_ylim(max(0.0, scenario.collision_distance - 0.16), upper)

        # Use a figure-level legend for edge identities so it is guaranteed to
        # remain visible after tight PDF cropping.
        fig.legend(
            handles=edge_handles,
            title=r"edge $\varepsilon_k=(i,j)$",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=6,
            columnspacing=1.25,
            handlelength=2.3,
            handletextpad=0.55,
            fontsize=9.6,
            title_fontsize=10.2,
            frameon=False,
        )

        style_handles = [
            Line2D([0], [0], color="0.25", linewidth=2.3, linestyle="-", label="established"),
            Line2D(
                [0], [0], color="0.25", linewidth=2.3,
                linestyle=(0, (5.5, 2.5)), label="prospective",
            ),
            Line2D(
                [0], [0], color="0.25", linewidth=1.75,
                linestyle=(0, (4.5, 1.8, 1.2, 1.8)),
                label=r"$d_{\max}+\rho_k$",
            ),
        ]
        ax.legend(
            handles=style_handles,
            loc="upper left",
            ncol=3,
            handlelength=2.7,
            columnspacing=1.3,
            fontsize=9.3,
        )

        fig.subplots_adjust(left=0.105, right=0.99, bottom=0.285, top=0.90)
        return _save(fig, output_dir, "paper_edge_distances_detailed")



def paper_open_system_counts(
    result: SimulationResult,
    output_dir: Path,
) -> Path:
    """Paper-quality summary of robot lifecycle and graph-size evolution."""
    with paper_style():
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.65),
            gridspec_kw={"height_ratios": [1.08, 0.92], "hspace": 0.10},
        )

        active_count = result.active.sum(axis=1)
        staging_count = result.staging.sum(axis=1)
        returning_count = result.returning.sum(axis=1)
        parked_count = result.parked.sum(axis=1)
        established_count = np.asarray(
            [len(edges) for edges in result.edge_history], dtype=int
        )
        prospective_count = np.asarray(
            [len(edges) for edges in result.prospective_history], dtype=int
        )

        blue, orange, yellow, purple, green = MATLAB_COLORS[:5]

        axes[0].step(
            result.time, active_count, where="post", color=blue, linewidth=2.50,
            label=r"OMRS robots $|\mathcal V_\phi|$",
        )
        axes[0].step(
            result.time, staging_count, where="post", color=yellow, linewidth=2.25,
            label="joining robots",
        )
        axes[0].step(
            result.time, returning_count, where="post", color=green, linewidth=2.25,
            label="returning robots",
        )
        axes[0].step(
            result.time, parked_count, where="post", color=purple, linewidth=2.25,
            label="charging robots",
        )
        axes[0].set_ylabel("robots")
        max_robots = int(
            max(
                np.max(active_count),
                np.max(staging_count),
                np.max(returning_count),
                np.max(parked_count),
            )
        )
        axes[0].set_ylim(-0.25, max_robots + 0.55)
        axes[0].set_yticks(np.arange(0, max_robots + 1, 1))
        axes[0].legend(
            ncol=2, loc="upper left", columnspacing=1.25,
            handlelength=2.6, handletextpad=0.55,
        )

        axes[1].step(
            result.time, established_count, where="post", color=green, linewidth=2.50,
            label=r"established $|\mathcal E_\phi|$",
        )
        axes[1].step(
            result.time, prospective_count, where="post", color=orange, linewidth=2.50,
            label=r"prospective $|\mathcal E_\phi^{\mathrm p}|$",
        )
        axes[1].set_ylabel("edges")
        max_edges = int(max(np.max(established_count), np.max(prospective_count)))
        axes[1].set_ylim(-0.25, max_edges + 0.55)
        edge_tick_step = 1 if max_edges <= 6 else 2
        axes[1].set_yticks(np.arange(0, max_edges + 1, edge_tick_step))
        axes[1].legend(
            ncol=2, loc="upper left", columnspacing=1.25,
            handlelength=2.6, handletextpad=0.55,
        )

        _add_event_lines(axes, result, labels=True)
        for index, ax in enumerate(axes):
            _finish_time_axis(ax, result, xlabel=index == len(axes) - 1)
            if index < len(axes) - 1:
                ax.tick_params(labelbottom=False)

        fig.subplots_adjust(left=0.105, right=0.99, bottom=0.115, top=0.90)
        return _save(fig, output_dir, "open_system_counts")

def save_paper_plots(
    result: SimulationResult,
    scenario: Scenario,
    output_dir: Path,
) -> list[Path]:
    """Generate the manuscript-oriented subset of simulation figures."""
    return [
        paper_formation_performance(result, output_dir),
        paper_constraint_summary(result, scenario, output_dir),
        paper_edge_distances_detailed(result, scenario, output_dir),
        paper_open_system_counts(result, output_dir),
    ]
