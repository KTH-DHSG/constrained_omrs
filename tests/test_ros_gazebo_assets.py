from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from constrained_omrs.scenario import default_open_formation_scenario


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "worlds" / "open_team.sdf"
RECORDING_WORLD = ROOT / "worlds" / "open_team_recording.sdf"
MODEL = ROOT / "models" / "omrs_quadrotor" / "model.sdf"
BRIDGE = ROOT / "config" / "bridge.yaml"
LAUNCH = ROOT / "launch" / "open_team_gazebo.launch.py"


def _parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def test_gazebo_sdf_assets_are_well_formed_xml() -> None:
    assert _parse(WORLD).tag == "sdf"
    assert _parse(RECORDING_WORLD).tag == "sdf"
    assert _parse(MODEL).tag == "sdf"


def test_recording_world_reduces_render_and_physics_load() -> None:
    root = _parse(RECORDING_WORLD)
    world = root.find("world")
    assert world is not None
    physics = world.find("physics")
    assert physics is not None
    assert float(physics.findtext("max_step_size")) == 0.002
    light = world.find("light[@name='sun']")
    assert light is not None
    assert light.findtext("cast_shadows") == "false"


def test_world_contains_seven_vehicle_instances_on_charging_pads() -> None:
    root = _parse(WORLD)
    world = root.find("world")
    assert world is not None

    includes = {
        include.findtext("name"): include
        for include in world.findall("include")
        if include.findtext("uri") == "model://omrs_quadrotor"
    }
    assert set(includes) == {f"drone_{i}" for i in range(7)}

    scenario = default_open_formation_scenario(3)
    assert scenario.charging_area is not None
    for robot in range(7):
        pose_text = includes[f"drone_{robot}"].findtext("pose")
        assert pose_text is not None
        pose = np.fromstring(pose_text, sep=" ")
        np.testing.assert_allclose(
            pose[:3], scenario.charging_area.pad_positions[robot], atol=1e-9
        )


def test_every_vehicle_has_four_motor_systems_and_one_odometry_publisher() -> None:
    root = _parse(WORLD)
    world = root.find("world")
    assert world is not None

    includes = [
        include
        for include in world.findall("include")
        if include.findtext("uri") == "model://omrs_quadrotor"
    ]
    assert len(includes) == 7

    for robot, include in enumerate(includes):
        plugins = include.findall("plugin")
        motor_plugins = [
            plugin
            for plugin in plugins
            if plugin.attrib.get("name") == "gz::sim::systems::MulticopterMotorModel"
        ]
        odom_plugins = [
            plugin
            for plugin in plugins
            if plugin.attrib.get("name") == "gz::sim::systems::OdometryPublisher"
        ]

        assert len(motor_plugins) == 4
        assert len(odom_plugins) == 1
        assert sorted(int(plugin.findtext("actuator_number")) for plugin in motor_plugins) == [
            0,
            1,
            2,
            3,
        ]
        assert all(
            plugin.findtext("robotNamespace") == f"drone_{robot}"
            for plugin in motor_plugins
        )
        assert all(
            plugin.findtext("commandSubTopic") == "command/motor_speed"
            for plugin in motor_plugins
        )
        assert odom_plugins[0].findtext("odom_topic") == f"/model/drone_{robot}/odometry"


def test_motor_model_matches_python_vehicle_parameters() -> None:
    root = _parse(WORLD)
    world = root.find("world")
    assert world is not None
    first_vehicle = next(
        include
        for include in world.findall("include")
        if include.findtext("name") == "drone_0"
    )
    motors = [
        plugin
        for plugin in first_vehicle.findall("plugin")
        if plugin.attrib.get("name") == "gz::sim::systems::MulticopterMotorModel"
    ]
    assert len(motors) == 4

    expected_motor_constant = 7.0 / (900.0**2)
    for motor in motors:
        assert float(motor.findtext("timeConstantUp")) == 0.045
        assert float(motor.findtext("timeConstantDown")) == 0.045
        assert float(motor.findtext("maxRotVelocity")) == 900.0
        assert np.isclose(float(motor.findtext("motorConstant")), expected_motor_constant)
        assert float(motor.findtext("momentConstant")) == 0.018

    # With the chosen rotor ordering, opposite rotors spin in the same direction.
    directions = [motor.findtext("turningDirection") for motor in motors]
    assert directions == ["ccw", "cw", "ccw", "cw"]


def test_bridge_has_clock_and_one_odom_and_motor_bridge_per_vehicle() -> None:
    text = BRIDGE.read_text()
    assert 'ros_topic_name: "/clock"' in text
    assert 'gz_type_name: "gz.msgs.Clock"' in text

    for robot in range(7):
        assert text.count(f'ros_topic_name: "/drone_{robot}/odom"') == 1
        assert text.count(f'gz_topic_name: "/model/drone_{robot}/odometry"') == 1
        assert text.count(
            f'ros_topic_name: "/drone_{robot}/command/motor_speed"'
        ) == 1
        assert text.count(
            f'gz_topic_name: "/drone_{robot}/command/motor_speed"'
        ) == 1

    assert text.count('ros_type_name: "actuator_msgs/msg/Actuators"') == 7
    assert text.count('gz_type_name: "gz.msgs.Actuators"') == 7


def test_launch_passes_headless_as_a_gazebo_flag_not_ros_gz_argument() -> None:
    text = LAUNCH.read_text()
    assert '"headless": LaunchConfiguration("headless")' not in text
    assert "headless_flag" in text
    assert "-s " in text
    assert '"gz_args"' in text
    assert '"recording_mode"' in text
    assert "open_team_recording.sdf" in text
    assert "selected_world" in text


def test_ros_odometry_subscription_uses_sensor_data_qos() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    assert "from rclpy.qos import qos_profile_sensor_data" in controller
    assert "qos_profile_sensor_data," in controller
    assert "lambda msg, robot=robot: self._odom_callback(robot, msg),\n                    20," not in controller


def test_quadrotor_sdf_does_not_use_unsupported_canonical_link_tag() -> None:
    text = (ROOT / "models" / "omrs_quadrotor" / "model.sdf").read_text()
    assert "<canonical_link>" not in text


def test_motor_plugins_reference_links_and_joints_local_to_included_model() -> None:
    """MulticopterMotorModel resolves names through Model::JointByName/LinkByName.

    The included quadrotor has direct child joints / links named rotor_i_joint and
    rotor_i.  Prefixing them with the include instance name (e.g.
    drone_0/rotor_0_joint) makes lookup fail silently in Gazebo Sim 8 and the
    motor system never applies thrust.
    """
    model_root = _parse(MODEL)
    model = model_root.find("model")
    assert model is not None
    local_joints = {joint.attrib["name"] for joint in model.findall("joint")}
    local_links = {link.attrib["name"] for link in model.findall("link")}

    world_root = _parse(WORLD)
    world = world_root.find("world")
    assert world is not None
    for include in world.findall("include"):
        if include.findtext("uri") != "model://omrs_quadrotor":
            continue
        for motor in include.findall("plugin"):
            if motor.attrib.get("name") != "gz::sim::systems::MulticopterMotorModel":
                continue
            assert motor.findtext("jointName") in local_joints
            assert motor.findtext("linkName") in local_links


def test_rotor_joints_have_effort_and_velocity_limits_for_velocity_control() -> None:
    model_root = _parse(MODEL)
    model = model_root.find("model")
    assert model is not None
    joints = [joint for joint in model.findall("joint") if joint.attrib["name"].startswith("rotor_")]
    assert len(joints) == 4
    for joint in joints:
        limit = joint.find("axis/limit")
        assert limit is not None
        assert float(limit.findtext("effort")) >= 1.0e6
        assert float(limit.findtext("velocity")) >= 1.0e6


def test_gazebo_controller_has_explicit_takeoff_and_correct_yaw_wrench_signs() -> None:
    text = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    assert '"spool"' in text
    assert '"takeoff"' in text
    assert '"deploy"' in text
    assert "starting OMRS mission clock" in text
    assert "rotor_spin_directions=(-1, 1, -1, 1)" in text


def test_gazebo_service_flight_uses_conservative_damped_defaults() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    assert 'self.declare_parameter("service_maximum_speed", 0.45)' in controller
    assert 'self.declare_parameter("service_maximum_acceleration", 0.90)' in controller
    assert 'self.declare_parameter("service_position_gain", 0.45)' in controller
    assert 'self.declare_parameter("service_velocity_gain", 2.20)' in controller
    assert 'self.declare_parameter("service_settle_time", 1.20)' in controller
    assert 'self.declare_parameter("attitude_gain", 1.8)' in controller
    assert 'self.declare_parameter("angular_velocity_gain", 0.50)' in controller
    assert "Preflight convergence --" in controller
    assert "service_maximum_speed: 0.45" in config
    assert "service_maximum_acceleration: 0.90" in config
    assert "attitude_gain: 1.8" in config


def test_service_point_controller_brakes_near_target() -> None:
    scenario = default_open_formation_scenario(3)
    charging = scenario.charging_area
    assert charging is not None
    target = np.zeros(3)
    # With a small positive position error but a larger positive velocity, the
    # service controller should command acceleration opposite to the motion.
    command = charging.point_acceleration_command(
        np.array([-0.10, 0.0, 0.0]),
        np.array([0.20, 0.0, 0.0]),
        target,
        maximum_speed=0.45,
        maximum_acceleration=0.90,
        position_gain=0.45,
        velocity_gain=2.20,
    )
    assert command[0] < 0.0
    assert np.linalg.norm(command) <= 0.90 + 1e-12


def test_gazebo_formation_controller_is_softened_and_command_limited() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    for text in (controller, config):
        assert "formation_k1" in text
        assert "formation_k2" in text
        assert "formation_k3" in text
        assert "formation_max_virtual_speed" in text
        assert "formation_maximum_acceleration" in text
        assert "formation_acceleration_rate_limit" in text
    assert "_condition_formation_acceleration" in controller
    assert "formation_k1: 0.30" in config
    assert "formation_maximum_acceleration: 0.85" in config
    assert "formation_acceleration_rate_limit: 0.0" in config


def test_mission_completion_generates_outputs_and_launch_shuts_down() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    launch = LAUNCH.read_text()
    assert "MISSION COMPLETE" in controller
    assert "_completion_ready" in controller
    assert "GazeboMissionRecorder" in controller
    assert "gazebo_mission.gif" in controller
    assert "stop_on_mission_complete: true" in config
    assert 'output_directory: "package_outputs"' in config
    assert "_source_package_outputs_directory" in (ROOT / "src" / "constrained_omrs_ros" / "gazebo_results.py").read_text()
    assert "OnProcessExit" in launch
    assert "Shutdown(reason=\"OMRS controller finished\")" in launch


def test_native_gazebo_edge_marker_support_is_present() -> None:
    markers = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_markers.py").read_text()
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    assert "/marker_array" in markers
    assert "LINE_LIST" in markers
    assert "omrs_established" in markers
    assert "omrs_prospective" in markers
    assert "self._Marker.TEXT" not in markers
    assert "update_snapshot" in markers
    assert "omrs-gazebo-marker-worker" in controller
    assert "_queue_native_gazebo_markers" in controller
    assert "python3-gz-transport13" in (ROOT / "README.md").read_text()
    assert "gazebo_edge_markers: true" in config
    assert "marker_rate_hz: 10.0" in config


def test_gazebo_controller_enables_collision_barriers_for_non_graph_pairs() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    assert "nearby_collision_pairs" in controller
    assert "_collision_pairs_for_handover_mode" in controller
    assert "excluded_edges=excluded" in controller
    assert 'self.declare_parameter("formation_collision_weight", 0.02)' in controller
    assert "formation_collision_weight: 0.02" in config


def test_native_gazebo_markers_use_boolean_response_and_visible_cylinders() -> None:
    markers = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_markers.py").read_text()
    assert "boolean_pb2 import Boolean" in markers
    assert '"/marker_array"' in markers
    assert "self._Boolean" in markers
    assert "self._Marker.CYLINDER" in markers
    assert "Native Gazebo interaction-edge visualization is active" in (
        ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py"
    ).read_text()


def test_gazebo_join_uses_rendezvous_capture_before_prospective_edges() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    assert '"join_approach"' in controller
    assert "_join_capture_ready" in controller
    assert "rendezvous capture reached; enabling prospective edges" in controller
    assert 'self.declare_parameter("join_capture_margin", 0.50)' in controller
    assert 'self.declare_parameter("join_capture_lower_margin", 0.20)' in controller
    assert "join_capture_margin: 0.50" in config
    assert "join_capture_lower_margin: 0.20" in config
    assert "rho_initial_margin: 0.22" in config
    assert "rho_contraction_rate: 0.10" in config
    assert "activation_distance_margin: 0.08" in config
    assert "activation_outward_rate_tolerance: 0.08" in config


def test_gazebo_edges_are_thick_dashed_green_with_orange_prospective_edges() -> None:
    marker_helper = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_markers.py").read_text()
    assert "established_dash_count = 8" in marker_helper
    assert "rgba=(0.03, 0.40, 0.08, 0.98)" in marker_helper
    assert "diameter=0.075" in marker_helper
    assert "rgba=(0.95, 0.45, 0.04, 0.98)" in marker_helper
    assert "diameter=0.060" in marker_helper


def test_native_gazebo_markers_avoid_unsupported_ogre2_text_type() -> None:
    marker_helper = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_markers.py").read_text()
    assert "self._Marker.TEXT" not in marker_helper
    assert '"omrs_edge_labels"' not in marker_helper


def test_gazebo_uses_linear_relaxation_and_zero_rho_activation() -> None:
    reconfiguration = (ROOT / "src" / "constrained_omrs" / "reconfiguration.py").read_text()
    controller = (ROOT / "src" / "constrained_omrs" / "controller.py").read_text()
    assert "-cfg.rho_contraction_rate" in reconfiguration
    assert "state.rho <= 1e-9" in reconfiguration
    assert "edge_activation_outward_rate_tolerance" in reconfiguration
    assert "response_weight_rates" in controller
    assert "R_dot @ gradients" in controller


def test_join_candidate_only_prospective_edges_are_distinct_from_bridge_edges() -> None:
    reconfiguration = (ROOT / "src" / "constrained_omrs" / "reconfiguration.py").read_text()
    assert 'if state.purpose == "join"' in reconfiguration
    assert "responders = (self._pending_robot,)" in reconfiguration
    assert 'state.purpose == "bridge"' in (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()


def test_gazebo_join_uses_whole_controller_homotopy() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    assert "_blend_high_level_outputs" in controller
    assert "_evaluate_formation_controller" in controller
    assert 'controller_interactions("pre")' in controller
    assert 'controller_interactions("full")' in controller
    assert "JOIN HOMOTOPY" in controller
    assert "join_handover_duration: 6.0" in config


def test_barrier_domain_errors_report_the_interaction_edge() -> None:
    controller = (ROOT / "src" / "constrained_omrs" / "controller.py").read_text()
    assert "interaction_edge={interaction.edge}" in controller


def test_interrupted_runs_save_logs_plots_and_support_fixed_team_debug_mode() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    results = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_results.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    launch = LAUNCH.read_text()
    assert "save_interrupted_results" in controller
    assert 'status="interrupted"' in controller
    assert "robot_diagnostics.csv" in results
    assert "edge_diagnostics.csv" in results
    assert "controller_exceptions.csv" in controller
    assert "fixed_team_only: false" in config
    assert '"fixed_team_only"' in launch
    assert "ParameterValue" in launch


def test_gazebo_join_capture_uses_relative_vector_error_and_real_prospective_phase() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    reconfiguration = (ROOT / "src" / "constrained_omrs" / "reconfiguration.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()
    assert 'self.declare_parameter("join_capture_error_tolerance", 1.10)' in controller
    assert "actual_relative - desired_relative" in controller
    assert "self.scenario.collision_distance + self.join_capture_lower_margin" in controller
    assert "join_capture_error_tolerance: 1.10" in config
    assert "join_capture_speed_tolerance: 0.45" in config
    assert "join_prospective_min_duration: 2.5" in config
    assert "join_activation_error_tolerance: 0.45" in config
    assert "state.assigned_at" in reconfiguration
    assert "cfg.join_prospective_min_duration" in reconfiguration
    assert "cfg.join_activation_error_tolerance" in reconfiguration


def test_second_join_target_is_compatible_with_gazebo_activation_margin() -> None:
    from constrained_omrs import default_open_formation_scenario

    scenario = default_open_formation_scenario(dimension=3)
    robot = scenario.join_requests[1].robot
    desired = scenario.desired_positions
    margin = 0.08
    feasible = 0
    for j in range(robot):
        d = float(np.linalg.norm(desired[robot] - desired[j]))
        if (
            scenario.collision_distance + margin < d
            and d < scenario.sensing_distance - margin
        ):
            feasible += 1
    assert feasible >= scenario.join_requests[1].num_edges


def test_native_gazebo_markers_clear_only_when_topology_changes() -> None:
    markers = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_markers.py").read_text()
    assert "self._active_namespaces" in markers
    assert "topology_changed" in markers
    assert "established_edges != self._last_established_edges" in markers
    assert "prospective_edges != self._last_prospective_edges" in markers
    assert "self._active_namespaces = current_namespaces" in markers


def test_gazebo_uses_smooth_damping_pdf_outputs_and_graceful_recovery() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    core = (ROOT / "src" / "constrained_omrs" / "controller.py").read_text()
    plotting = (ROOT / "src" / "constrained_omrs" / "plotting.py").read_text()
    results = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_results.py").read_text()
    config = (ROOT / "config" / "gazebo_controller.yaml").read_text()

    assert "max_edge_damping" in core
    assert "max_local_damping" in core
    assert "edge_velocity_error" in core
    assert "formation_max_edge_damping: 0.22" in config
    assert "formation_max_local_damping: 0.38" in config
    assert 'fig.savefig(path.with_suffix(".pdf")' in plotting
    assert "FFMpegWriter" in plotting
    assert "gazebo_mission.mp4" in results
    assert "return_to_charge_on_completion: true" in config
    assert "_begin_post_mission_recovery" in controller
    assert "all robots are parked" in controller


def test_runtime_performance_diagnostics_are_saved() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    assert "runtime_performance.csv" in controller
    assert "callback_wall_ms" in controller
    assert "local_real_time_factor" in controller
    assert "sim_period_ratio" in controller


def test_gazebo_odometry_freshness_uses_source_simulation_timestamp() -> None:
    controller = (ROOT / "src" / "constrained_omrs_ros" / "gazebo_controller.py").read_text()
    assert "last_state_time" in controller
    assert "msg.header.stamp" in controller
    assert "state_time + 1e-9 < state.last_state_time" in controller
    assert "state.last_state_time = state_time" in controller
