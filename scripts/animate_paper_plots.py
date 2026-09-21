#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from constrained_omrs.paper_animation import PaperAnimationConfig, save_paper_animations
from constrained_omrs_ros.gazebo_results import load_saved_gazebo_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the three manuscript-style result plots as progressive MP4 animations "
            "from an already saved Gazebo run. Gazebo does not need to be rerun."
        )
    )
    parser.add_argument(
        "run_directory",
        nargs="?",
        default=None,
        help=(
            "Saved run directory containing gazebo_history.npz, summary.json and "
            "edge_diagnostics.csv. If omitted, outputs/latest_run.txt is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed relative to simulation time. Keep 1.0 for Gazebo synchronization.",
    )
    parser.add_argument("--bitrate-kbps", type=int, default=6500)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("edge_distances", "formation_performance", "open_system_counts"),
        default=None,
        help="Render only the selected paper plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, scenario, run_dir = load_saved_gazebo_run(args.run_directory)
    output_dir = args.output_dir or (run_dir / "paper_animations")
    config = PaperAnimationConfig(
        fps=args.fps,
        dpi=args.dpi,
        playback_speed=args.speed,
        bitrate_kbps=args.bitrate_kbps,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    selected = tuple(args.only) if args.only else (
        "edge_distances",
        "formation_performance",
        "open_system_counts",
    )

    print(f"Loading Gazebo run: {run_dir}")
    print(f"Rendering paper plot animations to: {output_dir.resolve()}")
    print(
        f"Time base: {config.playback_speed:g}x simulation time, "
        f"{config.fps} fps, {config.dpi} dpi"
    )
    paths = save_paper_animations(
        result,
        scenario,
        output_dir,
        config=config,
        selected=selected,
    )
    print("Paper plot animations complete:")
    for path in paths:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
