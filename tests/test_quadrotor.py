from dataclasses import replace

import numpy as np

from constrained_omrs import (
    ControllerConfig,
    GeometricQuadrotorController,
    QuadrotorConfig,
    SmoothBacksteppingController,
    default_open_formation_scenario,
    simulate,
)


def test_geometric_controller_hover_command_is_physical() -> None:
    config = QuadrotorConfig()
    controller = GeometricQuadrotorController(config)
    output = controller.evaluate(
        np.eye(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )

    np.testing.assert_allclose(output.thrust, config.mass * config.gravity, atol=1e-12)
    np.testing.assert_allclose(output.torque, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(output.realized_acceleration, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(output.desired_rotation, np.eye(3), atol=1e-12)


def test_short_6dof_simulation_keeps_rotations_on_so3() -> None:
    scenario = replace(default_open_formation_scenario(3), duration=0.5)
    controller = SmoothBacksteppingController(
        num_robots=scenario.num_robots,
        dimension=3,
        collision_distance=scenario.collision_distance,
        sensing_distance=scenario.sensing_distance,
        config=ControllerConfig(k1=0.5, k2=0.5, k3=2.0, max_virtual_speed=0.6),
    )
    result = simulate(
        scenario,
        controller,
        dynamics="quadrotor",
        quadrotor_config=QuadrotorConfig(),
    )

    assert result.attitudes is not None
    assert result.thrusts is not None
    assert result.torques is not None
    assert result.attitude_errors is not None
    for rotation in result.attitudes[:, scenario.initial_active].reshape(-1, 3, 3):
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=2e-10)
        assert np.linalg.det(rotation) > 0.999999999

    config = result.quadrotor_config
    assert config is not None
    assert np.max(result.thrusts) <= config.maximum_total_thrust + 1e-12
    assert np.max(np.linalg.norm(result.torques, axis=2)) <= config.maximum_torque + 1e-12


def test_rotor_allocator_reproduces_hover_with_equal_rotor_thrusts() -> None:
    from constrained_omrs import QuadrotorRotorActuator

    config = QuadrotorConfig()
    actuator = QuadrotorRotorActuator(config)
    allocation = actuator.allocate(config.mass * config.gravity, np.zeros(3))

    np.testing.assert_allclose(
        allocation.commands,
        np.full(4, config.hover_rotor_thrust),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actuator.wrench(allocation.commands),
        np.array([config.mass * config.gravity, 0.0, 0.0, 0.0]),
        atol=1e-10,
    )
    assert not allocation.saturated


def test_first_order_rotor_dynamics_move_toward_command() -> None:
    from constrained_omrs import QuadrotorRotorActuator

    config = QuadrotorConfig(motor_time_constant=0.05)
    actuator = QuadrotorRotorActuator(config)
    thrusts = np.full(4, 2.0)
    commands = np.full(4, 4.0)
    next_thrusts = actuator.integrate(thrusts, commands, 0.01)

    assert np.all(next_thrusts > thrusts)
    assert np.all(next_thrusts < commands)
    np.testing.assert_allclose(next_thrusts, np.full(4, 2.4), atol=1e-12)


def test_short_rotor_level_simulation_respects_individual_bounds() -> None:
    scenario = replace(default_open_formation_scenario(3), duration=0.5)
    controller = SmoothBacksteppingController(
        num_robots=scenario.num_robots,
        dimension=3,
        collision_distance=scenario.collision_distance,
        sensing_distance=scenario.sensing_distance,
        config=ControllerConfig(k1=0.5, k2=0.5, k3=2.0, max_virtual_speed=0.6),
    )
    config = QuadrotorConfig()
    result = simulate(
        scenario,
        controller,
        dynamics="quadrotor_rotor",
        quadrotor_config=config,
    )

    assert result.rotor_thrusts is not None
    assert result.rotor_thrust_commands is not None
    assert result.rotor_allocation_saturated is not None
    assert np.min(result.rotor_thrusts) >= config.minimum_rotor_thrust - 1e-12
    assert np.max(result.rotor_thrusts) <= config.maximum_rotor_thrust + 1e-12
    assert np.min(result.rotor_thrust_commands) >= config.minimum_rotor_thrust - 1e-12
    assert np.max(result.rotor_thrust_commands) <= config.maximum_rotor_thrust + 1e-12
