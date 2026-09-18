from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from constrained_omrs import (
    ControllerConfig,
    QuadrotorConfig,
    SmoothBacksteppingController,
    default_open_formation_scenario,
    simulate,
)
from constrained_omrs.plotting import save_animation, save_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate prospective-edge reconfiguration of a constrained OMRS."
    )
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--dynamics",
        choices=("point_mass", "quadrotor", "quadrotor_rotor"),
        default="point_mass",
        help=(
            "Plant model. The default 'point_mass' is the double-integrator "
            "model considered in the paper; 'quadrotor' uses 6-DoF rigid-body "
            "dynamics with ideal wrench realization, and 'quadrotor_rotor' "
            "additionally includes bounded rotor allocation and first-order "
            "rotor dynamics."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--animation-fps", type=int, default=15)
    parser.add_argument(
        "--animation-stride",
        type=int,
        default=None,
        help=(
            "Use every Nth integration sample as an animation frame. "
            "Defaults to 10 for the 6-DoF quadrotor, 10 in 2-D, and 20 for "
            "the 3-D point-mass baseline."
        ),
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=5.0,
        help="Length of the trajectory tails shown in the 3-D animation.",
    )
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Keep the camera fixed in the 3-D animation instead of using a slow orbit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = default_open_formation_scenario(args.dimension)
    dynamics = args.dynamics
    if dynamics in ("quadrotor", "quadrotor_rotor") and scenario.dimension != 3:
        raise ValueError("The quadrotor dynamics require --dimension 3.")

    output_dir = args.output_dir or Path(
        f"outputs/open_formation_{args.dimension}d_{dynamics}"
    )

    if dynamics in ("quadrotor", "quadrotor_rotor"):
        # Use the same conservative physical-layer tuning as the Gazebo
        # validation. The paper-level point-mass simulation keeps the nominal
        # theoretical tuning below.
        scenario = replace(
            scenario,
            reconfiguration=replace(
                scenario.reconfiguration,
                rho_initial_margin=0.22,
                rho_contraction_rate=0.10,
                edge_activation_outward_rate_tolerance=0.08,
                join_handover_duration=6.0,
                join_prospective_min_duration=2.5,
                join_activation_error_tolerance=0.45,
            ),
        )
        controller_config = ControllerConfig(
            k1=0.30,
            k2=0.30,
            k3=1.25,
            max_virtual_speed=0.35,
            collision_weight=0.02,
            max_edge_damping=0.22,
            max_local_damping=0.38,
        )
    else:
        controller_config = ControllerConfig(
            k1=1.0, k2=0.8, k3=2.5, max_virtual_speed=0.8
        )
    controller = SmoothBacksteppingController(
        num_robots=scenario.num_robots,
        dimension=scenario.dimension,
        collision_distance=scenario.collision_distance,
        sensing_distance=scenario.sensing_distance,
        config=controller_config,
    )

    quadrotor_config = (
        QuadrotorConfig(
            attitude_gain=1.8,
            angular_velocity_gain=0.50,
            maximum_torque=0.70,
            maximum_tilt_deg=25.0,
        )
        if dynamics in ("quadrotor", "quadrotor_rotor")
        else None
    )
    result = simulate(
        scenario,
        controller,
        dynamics=dynamics,
        quadrotor_config=quadrotor_config,
    )
    paths = save_plots(result, scenario, output_dir, controller.config.max_virtual_speed)
    if args.animation:
        if args.animation_stride is not None:
            animation_stride = args.animation_stride
        elif dynamics in ("quadrotor", "quadrotor_rotor"):
            animation_stride = 10
        else:
            animation_stride = 10 if scenario.dimension == 2 else 20
        animation_kwargs = dict(
            fps=args.animation_fps,
            stride=animation_stride,
            tail_seconds=args.tail_seconds,
            rotate_3d_camera=not args.fixed_camera,
        )
        paths.append(
            save_animation(
                result,
                scenario,
                output_dir / f"open_formation_{args.dimension}d_{dynamics}.gif",
                **animation_kwargs,
            )
        )
        try:
            paths.append(
                save_animation(
                    result,
                    scenario,
                    output_dir / f"open_formation_{args.dimension}d_{dynamics}.mp4",
                    **animation_kwargs,
                )
            )
        except RuntimeError as exc:
            print(f"MP4 animation skipped: {exc}")

    print("Simulation complete.")
    print(f"Dimension: {scenario.dimension}-D")
    print(f"Plant: {dynamics}")
    if quadrotor_config is not None:
        print(
            "Quadrotor: "
            f"mass={quadrotor_config.mass:.2f} kg, "
            f"arm={quadrotor_config.arm_length:.2f} m, "
            f"propeller diameter={2.0 * quadrotor_config.propeller_radius:.2f} m, "
            f"max tilt={quadrotor_config.maximum_tilt_deg:.1f} deg"
        )
        controlled = result.controlled
        accel_error = np.linalg.norm(result.acceleration_tracking_error, axis=2)
        print(
            "Acceleration realization: "
            f"RMS={np.sqrt(np.mean(accel_error[controlled] ** 2)):.3f} m/s^2, "
            f"max={np.max(accel_error[controlled]):.3f} m/s^2"
        )
        if dynamics == "quadrotor_rotor":
            assert result.rotor_thrusts is not None
            assert result.rotor_allocation_saturated is not None
            flight = result.flight_controlled
            saturation_fraction = float(
                np.mean(result.rotor_allocation_saturated[flight])
            ) if np.any(flight) else 0.0
            print(
                "Rotor actuators: "
                f"tau={quadrotor_config.motor_time_constant:.3f} s, "
                f"Tmax={quadrotor_config.maximum_rotor_thrust:.2f} N/rotor, "
                f"allocation saturation={100.0 * saturation_fraction:.2f}%"
            )
    print(f"Final team size: {int(result.active[-1].sum())}")
    if scenario.charging_area is not None:
        print(
            f"Charging area: {int(result.parked[-1].sum())} parked, "
            f"{int(result.returning[-1].sum())} still returning at final time"
        )
    print(f"Final established edges: {result.edge_history[-1]}")
    print("Topology switches:")
    for event in result.switch_events:
        print(
            f"  t={event.time:6.2f} s  {event.kind:17s} "
            f"robot={event.robot}  +{list(event.edges_added)}  -{list(event.edges_removed)}"
        )
    print(
        "Edge-distance plot: each edge has its own curve; dashed segments are "
        "prospective and solid segments are established; dash-dotted curves "
        "show the edge-specific relaxed upper bounds d_max + rho_k."
    )
    print("Generated files:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
