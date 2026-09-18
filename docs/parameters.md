# Parameter reference

This page distinguishes the **paper-level/pure-Python scenario parameters** from the **Gazebo physical-realization parameters**. The authoritative sources are:

- `src/constrained_omrs/scenario.py`
- `src/constrained_omrs/controller.py`
- `src/constrained_omrs/quadrotor.py`
- `src/constrained_omrs/charging.py`
- `examples/run_open_formation.py`
- `config/gazebo_controller.yaml`

## Pure-Python CLI

| Parameter | Default | Description |
|---|---:|---|
| `--dimension` | `3` | Spatial dimension, `2` or `3` |
| `--dynamics` | `point_mass` | `point_mass`, `quadrotor`, or `quadrotor_rotor` |
| `--output-dir` | model-dependent path under `outputs/` | Output directory |
| `--animation` | off | Save animation |
| `--animation-fps` | `15` | Animation frame rate |
| `--animation-stride` | automatic | Integration-sample stride used for frames |
| `--tail-seconds` | `5.0` s | 3-D trajectory-tail duration |
| `--fixed-camera` | off | Disable the rotating 3-D camera |

## Controller configuration

`ControllerConfig` defines the high-level BLF/backstepping controller.

| Field | Class default | Paper-model example | Physical Python / Gazebo |
|---|---:|---:|---:|
| `k1` | `1.0` | `1.0` | `0.30` |
| `k2` | `1.0` | `0.8` | `0.30` |
| `k3` | `2.0` | `2.5` | `1.25` |
| `max_virtual_speed` | `0.8` m/s | `0.8` m/s | `0.35` m/s |
| `collision_weight` | `0.02` | `0.02` | `0.02` |
| `max_edge_damping` | `0.60` m/s² per edge | `0.60` | `0.22` |
| `max_local_damping` | `1.00` m/s² | `1.00` | `0.38` |

The two damping limits act on the smooth dissipative terms inside the backstepping law; they do not clip the final theoretical control vector.

## Default 3-D scenario

| Parameter | Value |
|---|---:|
| Number of robots | `7` |
| Initial active robots | `0,1,2,3,4` |
| Initial edges | `(0,1),(1,2),(2,3),(3,4)` |
| Join requests | robot `5` at `8 s`; robot `6` at `25 s` |
| Leave requests | robot `2` at `51 s`; robot `1` at `73 s`; robot `3` at `95 s` |
| `collision_distance = d_min` | `0.45 m` |
| `sensing_distance = d_max` | `1.65 m` |
| Duration | `123 s` |
| Integration step | `0.01 s` |

The desired positions used to define the relative formation are:

```text
robot 0: [-1.70, -0.55, 1.80] m
robot 1: [-0.90,  0.30, 2.35] m
robot 2: [ 0.00, -0.35, 2.90] m
robot 3: [ 0.90,  0.35, 2.25] m
robot 4: [ 1.70, -0.50, 2.75] m
robot 5: [ 1.95,  0.65, 3.15] m
robot 6: [-1.95,  0.60, 2.75] m
```

These coordinates are one representative of the desired formation; only relative displacements enter the formation objective.

## Pure-Python reconfiguration settings

The 3-D `Scenario` uses:

| Field | Value | Meaning |
|---|---:|---|
| `rho_initial_margin` | `0.18 m` | Additional protected margin when a prospective edge is created |
| `rho_contraction_rate` | `0.14 m/s` | Nominal constant decrease rate of `rho` |
| `lambda_rho` | `2.0 s^-1` | Feasibility-margin decay coefficient |
| `edge_activation_margin` | `0.08 m` | Interior margin protected from the relaxed upper boundary |
| `edge_activation_outward_rate_tolerance` | `0.10 m/s` | Outward radial-rate tolerance at activation |
| `min_switch_dwell_time` | `1.0 s` | Minimum interval between topology switches |
| `join_handover_duration` | `0.0 s` | Generic scenario default; physical Python modes override to `6.0 s` |
| `join_prospective_min_duration` | `0.0 s` | Generic scenario default; physical Python modes override to `2.5 s` |
| `join_activation_error_tolerance` | `inf` | Generic scenario default; physical Python modes override to `0.45 m` |

For `quadrotor` and `quadrotor_rotor`, `examples/run_open_formation.py` overrides the physical reconfiguration tuning to

```text
rho_initial_margin                  0.22 m
rho_contraction_rate                0.10 m/s
edge_activation_outward_rate_tol    0.08 m/s
join_handover_duration              6.0 s
join_prospective_min_duration       2.5 s
join_activation_error_tolerance     0.45 m
```

while retaining the scenario's `lambda_rho`, activation-distance margin, and switch dwell time.

## Quadrotor model

`QuadrotorConfig` defaults are:

| Field | Value |
|---|---:|
| Mass | `1.20 kg` |
| Inertia | `diag(0.018, 0.018, 0.032) kg m²` |
| Gravity | `9.81 m/s²` |
| Linear drag | `0.10` |
| Attitude gain | `4.0` class default; physical example uses `1.8` |
| Angular-velocity gain | `0.70` class default; physical example uses `0.50` |
| Maximum total thrust | `28 N` |
| Maximum torque | `1.50 N m` class default; physical example uses `0.70 N m` |
| Maximum tilt | `35 deg` class default; physical example uses `25 deg` |
| Arm length | `0.23 m` |
| Body radius | `0.075 m` |
| Body height | `0.07 m` |
| Propeller radius | `0.11 m` |
| Maximum rotor thrust | `7 N` |
| Motor time constant | `0.045 s` |
| Yaw moment / thrust | `0.018 m` |

The pure-Python rotor model represents rotor thrust as the actuator state and applies bounded weighted allocation before first-order motor dynamics.

## Charging/service configuration in the 3-D scenario

The scenario-level `ChargingAreaConfig` is:

| Field | Value |
|---|---:|
| `approach_height` | `0.90 m` |
| `approach_radius` | `0.50 m` |
| `maximum_speed` | `1.30 m/s` |
| `maximum_descent_speed` | `0.50 m/s` |
| `velocity_gain` | `2.2` |
| `maximum_acceleration` | `3.5 m/s²` |
| `landing_position_tolerance` | `0.10 m` |
| `landing_speed_tolerance` | `0.20 m/s` |
| `platform_padding` | `0.34 m` |

The physical pure-Python joining service layer deliberately uses more conservative internal limits matching the Gazebo validation: `0.45 m/s` maximum speed, `0.90 m/s²` maximum acceleration, position gain `0.45`, velocity gain `2.20`, and `0.85 m/s²` as the final formation-acceleration envelope.

## Gazebo controller parameters

When launched through `open_team_gazebo.launch.py`, `config/gazebo_controller.yaml` provides the effective values below.

### Timing and interfaces

| Parameter | Value | Description |
|---|---:|---|
| `use_sim_time` | `true` | Use Gazebo simulation clock |
| `control_rate_hz` | `100.0` | Controller timer rate |
| `marker_rate_hz` | `10.0` | Visualization update rate |
| `odom_timeout` | `0.25 s` | Stale-odometry threshold |
| `motor_constant` | `8.6419753e-6 N/(rad/s)^2` | Thrust coefficient used for speed/thrust conversion |
| `maximum_rotor_velocity` | `900 rad/s` | Motor-speed command bound |
| `diagnostic_log_period` | `1.0 s` | Periodic diagnostic logging |
| `switch_diagnostic_period` | `0.10 s` | Switch/handover diagnostic period |

### Service-flight layer

| Parameter | Value |
|---|---:|
| `motor_spool_time` | `1.0 s` |
| `initial_takeoff_stagger` | `0.75 s` |
| `takeoff_height` | `0.85 m` |
| `service_maximum_speed` | `0.45 m/s` |
| `service_maximum_acceleration` | `0.90 m/s²` |
| `service_position_gain` | `0.45` |
| `service_velocity_gain` | `2.20` |
| `service_position_tolerance` | `0.15 m` |
| `service_speed_tolerance` | `0.12 m/s` |
| `takeoff_settle_time` | `0.80 s` |
| `service_settle_time` | `1.20 s` |
| `join_capture_margin` | `0.50 m` |
| `join_capture_error_tolerance` | `0.75 m` |
| `join_capture_speed_tolerance` | `0.30 m/s` |

### High-level formation controller

| Parameter | Value |
|---|---:|
| `formation_k1` | `0.30` |
| `formation_k2` | `0.30` |
| `formation_k3` | `1.25` |
| `formation_max_virtual_speed` | `0.35 m/s` |
| `formation_collision_weight` | `0.02` |
| `formation_max_edge_damping` | `0.22 m/s²` per edge |
| `formation_max_local_damping` | `0.38 m/s²` |
| `formation_maximum_acceleration` | `0.85 m/s²` |
| `formation_acceleration_rate_limit` | `0.0` (disabled) |

### Prospective-edge and switching parameters

| Parameter | Value |
|---|---:|
| `rho_initial_margin` | `0.22 m` |
| `rho_contraction_rate` | `0.10 m/s` |
| `activation_distance_margin` | `0.08 m` |
| `activation_outward_rate_tolerance` | `0.08 m/s` |
| `join_handover_duration` | `6.0 s` |
| `join_prospective_min_duration` | `2.5 s` |
| `join_activation_error_tolerance` | `0.45 m` |

`activation_distance_margin` is written into the scenario's `edge_activation_margin` when the ROS node is initialized.

### Geometric/attitude realization

| Parameter | Value |
|---|---:|
| `attitude_gain` | `1.8` |
| `angular_velocity_gain` | `0.50` |
| `maximum_torque` | `0.70 N m` |
| `maximum_tilt_deg` | `25 deg` |

The remaining mass, inertia, geometry, thrust bounds and rotor time constant come from `QuadrotorConfig` and are matched to the Gazebo SDF model.

### Mission completion and output

| Parameter | Value |
|---|---:|
| `completion_position_tolerance` | `0.12` |
| `completion_velocity_tolerance` | `0.12` |
| `completion_settle_time` | `2.0 s` |
| `stop_on_mission_complete` | `true` |
| `return_to_charge_on_completion` | `true` |
| `save_results` | `true` |
| `save_animation` | `true` |
| `result_sample_rate_hz` | `20.0 Hz` |
| `animation_fps` | `15` |
| `animation_stride` | `2` |
| `output_directory` | `package_outputs` |
| `gazebo_edge_markers` | `true` |
| `fixed_team_only` | `false` |
| `save_animation_on_interrupt` | `true` |

`package_outputs` resolves to `<workspace>/src/constrained_omrs/outputs` when the package is built in a normal colcon workspace.

## Launch-file parameters

The following are launch arguments rather than ROS node parameters:

| Argument | Default | Meaning |
|---|---:|---|
| `headless` | `false` | Gazebo server without GUI |
| `verbosity` | `2` | Gazebo verbosity |
| `recording_mode` | `false` | Use lower-load recording world |
| `fixed_team_only` | `false` | Forwarded to the ROS node to disable membership changes |
