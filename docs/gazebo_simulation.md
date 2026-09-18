# ROS 2 / Gazebo simulation

The high-fidelity validation runs the open-team controller as a ROS 2 node while Gazebo Harmonic owns the rigid-body, contact, and motor physics.

## Target environment

The intended environment is:

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic through `ros_gz`
- Python 3 / NumPy / Matplotlib / Pillow

Install the required packages:

```bash
sudo apt update
sudo apt install \
  ros-jazzy-ros-gz \
  ros-jazzy-actuator-msgs \
  ros-jazzy-visualization-msgs \
  python3-colcon-common-extensions \
  python3-matplotlib \
  python3-pil \
  python3-gz-transport13 \
  python3-gz-msgs10
```

The Gazebo Transport Python bindings are only required for the native dashed interaction-edge visualization in the Gazebo GUI; the controller itself does not depend on those markers.

## Build

Assuming the repository is at `~/Code/omrs_ws/src/constrained_omrs`:

```bash
cd ~/Code/omrs_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select constrained_omrs
source install/setup.bash
```

## Launch modes

Normal GUI run:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py
```

Headless/server-only run:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py headless:=true
```

Lower-load screen-recording world:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py recording_mode:=true
```

Fixed-team controller check, with join/leave requests removed:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py fixed_team_only:=true
```

More Gazebo console output:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py verbosity:=4
```

Launch arguments can be combined, for example:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py \
  headless:=true \
  fixed_team_only:=true
```

## Launch arguments

| Argument | Default | Meaning |
|---|---:|---|
| `headless` | `false` | Run Gazebo server only (`-s`) |
| `verbosity` | `2` | Gazebo verbosity passed to `gz sim -v` |
| `recording_mode` | `false` | Use `open_team_recording.sdf` with lower rendering/physics load |
| `fixed_team_only` | `false` | Remove all join/leave requests and test only the initial formation |

All controller parameters are loaded from `config/gazebo_controller.yaml`. Most are read once during node construction, so change the YAML before launching rather than relying on runtime parameter updates.

## Mission lifecycle

All seven vehicles are physically initialized on the charging station. The OMRS mission clock starts only after robots `0--4` have completed spool-up, takeoff, and service-controller deployment to the prescribed initial airborne state.

The main mission then follows this sequence:

1. fixed-team formation convergence;
2. robot `5` join request at `8 s`;
3. robot `6` join request at `25 s`;
4. robot `2` departure request at `51 s`;
5. robot `1` departure request at `73 s`;
6. robot `3` departure request at `95 s`;
7. final settling and, when enabled, return of all remaining vehicles to their charging pads.

A joining drone first uses the service-flight controller to take off and approach the translation-aligned target formation. Prospective edges are assigned only after the capture checks are met. The candidate then runs prospective BLF control before admission, followed by the configured smooth whole-controller handover.

A connectivity-critical departure uses make-before-break: bridge edges are established first, bridge activation and robot removal are separate topology switches, and the departing vehicle then returns independently to its pad.

## Control stack

```text
FormationManager
  -> established/prospective interactions
  -> SmoothBacksteppingController
  -> desired translational acceleration
  -> physical acceleration envelope
  -> GeometricQuadrotorController
  -> thrust/torque
  -> QuadrotorRotorActuator allocation
  -> rotor angular-speed commands
  -> ros_gz_bridge
  -> Gazebo MulticopterMotorModel + rigid body
```

The Gazebo realization uses `d_min = 0.45 m` and `d_max = 1.65 m` from the 3-D scenario. The effective physical tuning is set in `config/gazebo_controller.yaml`; see [`parameters.md`](parameters.md).

## ROS/Gazebo interfaces

State feedback:

```text
/drone_0/odom ... /drone_6/odom     nav_msgs/msg/Odometry
```

Motor commands:

```text
/drone_0/command/motor_speed ...
/drone_6/command/motor_speed        actuator_msgs/msg/Actuators
```

The four `Actuators.velocity` entries are rotor angular-speed commands in rad/s.

Supervisory/visualization topics:

```text
/omrs/status                         std_msgs/msg/String
/omrs/markers                        visualization_msgs/msg/MarkerArray
```

`/omrs/status` reports active, staging, returning and parked robots, established/prospective edges, prospective relaxations, and collision-only physical pairs.

`/omrs/markers` can be displayed in RViz with fixed frame `world`. When `gazebo_edge_markers` is enabled and the Gazebo Transport Python bindings are available, established edges are also drawn directly in Gazebo as dashed dark-green segments and prospective edges as dashed orange segments.

## Result recording

With `save_results: true`, each run creates

```text
outputs/run_YYYYMMDD_HHMMSS/
```

and updates

```text
outputs/latest_run.txt
```

The run directory contains, depending on completion/interruption settings:

```text
gazebo_history.npz          raw/downsampled numerical history
summary.json                mission summary and switch events
robot_diagnostics.csv       per-robot state/control diagnostics
edge_diagnostics.csv        established/prospective edge diagnostics
controller_exceptions.csv   controller exceptions, if any
join_transition_log.csv     joining/handover diagnostics
runtime_performance.csv     callback timing / local real-time-factor diagnostics
gazebo_mission.gif          generated animation
gazebo_mission.mp4          generated MP4 when writer support is available
...                         standard plots generated by the plotting module
```

`runtime_performance.csv` is particularly useful when recording video. The controller uses simulation time, so a real-time factor below one does not by itself change the simulated dynamics; `sim_period_ratio` reveals actual missed/coalesced simulation-time controller periods.

## First-run checks

Check ROS state flow:

```bash
ros2 topic hz /drone_0/odom
ros2 topic echo /drone_0/odom --once
ros2 topic echo /omrs/status
```

Check Gazebo topics:

```bash
gz topic -l | grep drone_0
```

Check motor commands on both sides of the bridge:

```bash
ros2 topic echo /drone_0/command/motor_speed --once
gz topic -e -t /drone_0/command/motor_speed
```

## Troubleshooting

### Gazebo cannot resolve the quadrotor model

```bash
echo $GZ_SIM_RESOURCE_PATH
ros2 pkg prefix --share constrained_omrs
```

The installed package must contain `models/omrs_quadrotor`.

### No odometry reaches ROS

Compare the Gazebo and ROS topic lists and verify that `config/bridge.yaml` is installed and loaded by `parameter_bridge`.

### Strong yaw response at takeoff

Check the rotor turning-direction convention before changing gains. Gazebo's motor drag-torque convention is opposite the rotor turning direction; the allocator signs in the ROS node are selected to match the SDF model.

### Run is slow while screen recording

Use:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py recording_mode:=true
```

The recording world uses a `2 ms` physics step and disables shadows while retaining simulation-time control.
