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

A joining drone first uses the service-flight controller to take off and enter a coarse rendezvous region around its future neighbors. Prospective edges are then assigned before the service controller has completed the final geometric rendezvous, so the candidate-only prospective controller owns a visible part of the approach. The capture gate requires the selected future-edge distances to stay at least `join_capture_lower_margin` above `d_min`, while `FormationManager` initializes `rho` so the current distance is separated from the relaxed upper boundary by the configured activation and initial-relaxation margins. Admission is followed by the configured smooth whole-controller handover.

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

## Recording the final Gazebo video

For a publication / presentation recording, use the dedicated recording mode:

```bash
ros2 launch constrained_omrs open_team_gazebo.launch.py \
    video_recording:=true
```

This mode is deliberately interactive. Gazebo opens **paused**, loads a GUI
layout with the native **Video Recorder** button, and automatically enables a
Gazebo state log. The terminal prints a timestamped run directory, normally

```text
~/Videos/constrained_omrs/run_YYYYMMDD_HHMMSS/
```

Before starting the mission:

1. adjust the Gazebo camera if desired;
2. click the Video Recorder button and select MP4;
3. press **Play** in Gazebo.

The recorder uses Gazebo simulation time (`use_sim_time=true`), so moderate
wall-time / real-time-factor variation does not stretch the saved movie.
Lockstep recording is intentionally disabled to avoid unnecessarily slowing the
simulation.

At the end of the mission the controller does **not** immediately close Gazebo.
Instead it prints a `VIDEO CAPTURE READY` message. Stop the Video Recorder and,
in the save dialog, save the file inside the run directory printed by the
terminal. The controller polls that directory; once the `.mp4` or `.ogv` file
appears, it prints the exact video path and folder and then shuts down Gazebo
automatically.

The same run also stores a Gazebo simulation-state log below

```text
<run-directory>/gazebo_state/
```

which can later be replayed with

```bash
gz sim -r --playback <run-directory>/gazebo_state
```

The GUI video is the primary visual artifact. The state log is a useful backup
for replaying the physical vehicle motion.

The recording-related launch arguments are:

| Argument | Default | Meaning |
| --- | --- | --- |
| `video_recording` | `false` | Enables the complete paused video-capture workflow. |
| `start_paused` | `false` | Starts Gazebo paused independently of video recording. |
| `record_gazebo_log` | `false` | Enables Gazebo state logging; implied by `video_recording:=true`. |
| `video_output_directory` | timestamped folder under `~/Videos/constrained_omrs` | Folder in which the GUI video should be saved. |
| `gazebo_log_path` | `<video_output_directory>/gazebo_state` | Gazebo `--record-path` destination. |
| `recording_mode` | `false` | Selects the optional lower-load world; not required when the normal world runs comfortably in real time. |

## Paper-plot animations for the video

A completed Gazebo run can be post-processed into the same three plot groups used
in the paper results figure, with the manuscript Matplotlib style and MATLAB
ColorOrder:

- detailed edge distances, including established / prospective segments and
  the relaxed upper bound `d_max + rho_k`;
- formation position error, velocity error, and composite BLF;
- open-system robot and graph-edge counts.

The curves are progressively revealed in simulation time and switch-event lines
and labels appear when the corresponding event occurs.  The axes, line styles,
labels, colors, and limits are fixed from the complete run, so nothing jumps or
rescales while the video is playing.

After a run has been saved, render all three animations with

```bash
uv run --no-sync python scripts/animate_paper_plots.py
```

With no positional argument the script reads `outputs/latest_run.txt`.  An
explicit older run can instead be selected with

```bash
uv run --no-sync python scripts/animate_paper_plots.py \
  outputs/run_YYYYMMDD_HHMMSS
```

The default output is

```text
outputs/run_YYYYMMDD_HHMMSS/paper_animations/
├── paper_edge_distances_detailed.mp4
├── paper_formation_performance.mp4
└── open_system_counts.mp4
```

The default is 30 fps at real simulation speed (`--speed 1.0`), which makes the
clips directly synchronizable with the Gazebo recording in Blender.  If the
Gazebo footage is later accelerated, either accelerate all clips together in the
editor or pre-render the plots at the same factor, for example

```bash
uv run --no-sync python scripts/animate_paper_plots.py --speed 2
```

Useful rendering options are

```text
--fps 30
--dpi 180
--speed 1.0
--bitrate-kbps 6500
--start-time <seconds>
--end-time <seconds>
--only edge_distances formation_performance open_system_counts
```

The animation script uses the saved `gazebo_history.npz`, `summary.json`, and
`edge_diagnostics.csv`; Gazebo does not need to be rerun to change the video
rendering settings.
