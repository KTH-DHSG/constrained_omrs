from __future__ import annotations

from datetime import datetime
import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _launch_setup(context):
    package_share = get_package_share_directory("constrained_omrs")
    ros_gz_share = get_package_share_directory("ros_gz_sim")

    world = os.path.join(package_share, "worlds", "open_team.sdf")
    recording_world = os.path.join(package_share, "worlds", "open_team_recording.sdf")
    video_world = os.path.join(package_share, "worlds", "open_team_video.sdf")
    bridge_config = os.path.join(package_share, "config", "bridge.yaml")
    controller_config = os.path.join(package_share, "config", "gazebo_controller.yaml")

    headless = _as_bool(LaunchConfiguration("headless").perform(context))
    recording_mode = _as_bool(LaunchConfiguration("recording_mode").perform(context))
    fixed_team_only = _as_bool(LaunchConfiguration("fixed_team_only").perform(context))
    video_recording = _as_bool(LaunchConfiguration("video_recording").perform(context))
    start_paused = _as_bool(LaunchConfiguration("start_paused").perform(context)) or video_recording
    record_gazebo_log = (
        _as_bool(LaunchConfiguration("record_gazebo_log").perform(context))
        or video_recording
    )
    verbosity = LaunchConfiguration("verbosity").perform(context)

    video_output_directory = os.path.abspath(
        os.path.expanduser(LaunchConfiguration("video_output_directory").perform(context))
    )
    gazebo_log_path = os.path.abspath(
        os.path.expanduser(LaunchConfiguration("gazebo_log_path").perform(context))
    )

    if video_recording and headless:
        raise RuntimeError("video_recording:=true requires headless:=false.")

    if video_recording or record_gazebo_log:
        os.makedirs(video_output_directory, exist_ok=True)
    if record_gazebo_log:
        os.makedirs(os.path.dirname(gazebo_log_path), exist_ok=True)

    # Video capture uses a dedicated world with the same 1 ms physics as the
    # standard experiment, but a 25 Hz SceneBroadcaster and an embedded
    # Gazebo Video Recorder configured for simulation-time capture. Lockstep is
    # intentionally disabled: simulation-time timestamps already remove RTF stretching,
    # while avoiding Harmonic lockstep / GUI deadlocks on complex worlds.
    if video_recording:
        selected_world = video_world
    else:
        selected_world = recording_world if recording_mode else world

    gz_tokens: list[str] = []
    if headless:
        gz_tokens.append("-s")
    if not start_paused:
        gz_tokens.append("-r")
    gz_tokens.extend(["-v", verbosity])
    if record_gazebo_log:
        gz_tokens.extend(["--record-path", gazebo_log_path])
    gz_tokens.append(selected_world)
    gz_args = " ".join(shlex.quote(token) for token in gz_tokens)

    if video_recording:
        print("\n============================================================")
        print("OMRS GAZEBO VIDEO RECORDING MODE")
        print("============================================================")
        print("Gazebo will open PAUSED.")
        print("1. Adjust the camera if desired.")
        print("2. The Video Recorder button should already be visible; choose MP4.")
        print("3. Save the finished video in:")
        print(f"   {video_output_directory}")
        print("4. Press Play in Gazebo to start the experiment.")
        print("Video timing: simulation time, lockstep OFF, 25 FPS scene updates.")
        print("At mission completion, stop the recorder and save the file.")
        print("The controller will detect the saved video, print its path, and shut down.")
        if record_gazebo_log:
            print(f"Gazebo state log: {gazebo_log_path}")
        print("============================================================\n")
    elif start_paused:
        print("OMRS: Gazebo will start paused; press Play in the GUI when ready.")
    if record_gazebo_log and not video_recording:
        print(f"OMRS: Gazebo state recording enabled at: {gazebo_log_path}")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": gz_args,
            "on_exit_shutdown": "true",
        }.items(),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="open_team_bridge",
        output="screen",
        parameters=[{"config_file": bridge_config}],
    )

    controller = Node(
        package="constrained_omrs",
        executable="omrs_gazebo_controller",
        name="open_team_gazebo_controller",
        output="screen",
        parameters=[
            controller_config,
            {
                "fixed_team_only": fixed_team_only,
                # During video recording the controller remains alive after
                # the mission so the user has time to stop the GUI recorder.
                # It shuts itself down as soon as the saved MP4/OGV appears.
                "stop_on_mission_complete": not video_recording,
                "video_recording_mode": video_recording,
                "video_output_directory": video_output_directory if video_recording else "",
            },
        ],
    )

    controller_exit_shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=controller,
            on_exit=[Shutdown(reason="OMRS controller finished")],
        )
    )

    return [gazebo, bridge, controller, controller_exit_shutdown]


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("constrained_omrs")
    model_path = os.path.join(package_share, "models")

    current_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_path = (
        model_path
        if not current_resource_path
        else model_path + os.pathsep + current_resource_path
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_video_dir = os.path.join(
        os.path.expanduser("~/Videos"), "constrained_omrs", f"run_{timestamp}"
    )
    default_gazebo_log = os.path.join(default_video_dir, "gazebo_state")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run only the Gazebo server (no GUI).",
            ),
            DeclareLaunchArgument(
                "verbosity",
                default_value="2",
                description="Gazebo verbosity level.",
            ),
            DeclareLaunchArgument(
                "recording_mode",
                default_value="false",
                description=(
                    "Use the lower-load screen-recording world (2 ms physics step and "
                    "shadows disabled). Control still runs on Gazebo simulation time."
                ),
            ),
            DeclareLaunchArgument(
                "fixed_team_only",
                default_value="false",
                description=(
                    "Disable join / leave requests and run only the initial fixed-team "
                    "formation-control experiment."
                ),
            ),
            DeclareLaunchArgument(
                "start_paused",
                default_value="false",
                description=(
                    "Start Gazebo paused. video_recording:=true enables this automatically."
                ),
            ),
            DeclareLaunchArgument(
                "video_recording",
                default_value="false",
                description=(
                    "Prepare a Gazebo GUI video-recording run: start paused, use the "
                    "dedicated simulation-time video world (lockstep disabled), retain the controller "
                    "after mission completion, and record Gazebo state."
                ),
            ),
            DeclareLaunchArgument(
                "record_gazebo_log",
                default_value="false",
                description=(
                    "Record Gazebo simulation state. Enabled automatically by "
                    "video_recording:=true."
                ),
            ),
            DeclareLaunchArgument(
                "video_output_directory",
                default_value=default_video_dir,
                description=(
                    "Directory in which the Gazebo Video Recorder output should be saved. "
                    "A timestamped directory under ~/Videos/constrained_omrs is used by default."
                ),
            ),
            DeclareLaunchArgument(
                "gazebo_log_path",
                default_value=default_gazebo_log,
                description="Directory used by Gazebo --record-path.",
            ),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
            OpaqueFunction(function=_launch_setup),
        ]
    )
