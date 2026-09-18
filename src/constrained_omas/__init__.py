"""Constrained open multi-robot system simulation package."""

from .barrier import EdgeBarrier
from .charging import ChargingAreaConfig
from .controller import ControllerConfig, EdgeInteraction, SmoothBacksteppingController
from .quadrotor import (
    GeometricQuadrotorController,
    QuadrotorConfig,
    QuadrotorRotorActuator,
    RotorAllocationOutput,
)
from .reconfiguration import FormationManager, ProspectiveEdgeState, SwitchEvent
from .scenario import (
    JoinRequest,
    LeaveRequest,
    ReconfigurationConfig,
    Scenario,
    default_open_formation_scenario,
)
from .simulation import DynamicsModel, SimulationResult, simulate

__all__ = [
    "ChargingAreaConfig",
    "ControllerConfig",
    "DynamicsModel",
    "EdgeBarrier",
    "EdgeInteraction",
    "FormationManager",
    "GeometricQuadrotorController",
    "JoinRequest",
    "LeaveRequest",
    "ProspectiveEdgeState",
    "QuadrotorConfig",
    "QuadrotorRotorActuator",
    "ReconfigurationConfig",
    "RotorAllocationOutput",
    "Scenario",
    "SimulationResult",
    "SmoothBacksteppingController",
    "SwitchEvent",
    "default_open_formation_scenario",
    "simulate",
]

from .gazebo_adapter import (
    motor_speed_for_hover,
    odometry_body_twist_to_world_linear_velocity,
    quaternion_to_rotation_xyzw,
    rotor_thrusts_to_motor_speeds,
)

__all__ += [
    "motor_speed_for_hover",
    "odometry_body_twist_to_world_linear_velocity",
    "quaternion_to_rotation_xyzw",
    "rotor_thrusts_to_motor_speeds",
]
