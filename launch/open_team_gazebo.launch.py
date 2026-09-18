from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("constrained_omrs")
    ros_gz_share = get_package_share_directory("ros_gz_sim")

    world = os.path.join(package_share, "worlds", "open_team.sdf")
    recording_world = os.path.join(package_share, "worlds", "open_team_recording.sdf")
    bridge_config = os.path.join(package_share, "config", "bridge.yaml")
    controller_config = os.path.join(
        package_share, "config", "gazebo_controller.yaml"
    )
    model_path = os.path.join(package_share, "models")

    current_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_path = (
        model_path
        if not current_resource_path
        else model_path + os.pathsep + current_resource_path
    )

    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run only the Gazebo server (no GUI).",
    )
    verbosity_arg = DeclareLaunchArgument(
        "verbosity",
        default_value="2",
        description="Gazebo verbosity level.",
    )
    recording_mode_arg = DeclareLaunchArgument(
        "recording_mode",
        default_value="false",
        description=(
            "Use the lower-load screen-recording world (2 ms physics step and "
            "shadows disabled). Control still runs on Gazebo simulation time."
        ),
    )
    fixed_team_arg = DeclareLaunchArgument(
        "fixed_team_only",
        default_value="false",
        description=(
            "Disable join / leave requests and run only the initial fixed-team "
            "formation-control experiment."
        ),
    )

    # ros_gz_sim's gz_sim.launch.py accepts Gazebo command-line options through
    # gz_args.  In particular, server-only execution is requested by passing -s
    # to Gazebo itself rather than by forwarding a non-existent headless launch
    # argument to ros_gz_sim.
    headless_flag = PythonExpression(
        ["'-s ' if '", LaunchConfiguration("headless"), "'.lower() == 'true' else ''"]
    )
    selected_world = PythonExpression(
        [
            "'", recording_world, "' if '",
            LaunchConfiguration("recording_mode"),
            "'.lower() == 'true' else '", world, "'",
        ]
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": [
                headless_flag,
                "-r -v ",
                LaunchConfiguration("verbosity"),
                " ",
                selected_world,
            ],
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
                "fixed_team_only": ParameterValue(
                    LaunchConfiguration("fixed_team_only"), value_type=bool
                )
            },
        ],
    )

    # The controller exits automatically after mission completion and result
    # generation.  Propagate that exit to the Gazebo / bridge processes so a
    # successful experiment terminates cleanly without requiring Ctrl-C.
    controller_exit_shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=controller,
            on_exit=[Shutdown(reason="OMRS controller finished")],
        )
    )

    return LaunchDescription(
        [
            headless_arg,
            verbosity_arg,
            recording_mode_arg,
            fixed_team_arg,
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
            gazebo,
            bridge,
            controller,
            controller_exit_shutdown,
        ]
    )
