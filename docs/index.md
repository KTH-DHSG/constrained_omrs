# Documentation

This folder documents the implementation accompanying **“Safe Formation Control of Open Multi-Robot Systems with Connectivity-Preserving Reconfiguration.”**

The code is organized around one common formation/reconfiguration layer and several plant-fidelity levels. The double-integrator simulation is the closest implementation of the model analyzed in the paper; the quadrotor, rotor-level, and Gazebo simulations progressively add physical realization layers without changing the basic open-team coordination objective.

## Start here

| Document | Contents |
|---|---|
| [`architecture.md`](architecture.md) | Software structure, controller stack, prospective edges, joining/removal logic |
| [`python_simulation.md`](python_simulation.md) | Installation, CLI commands, dynamics modes, outputs, programmatic API |
| [`gazebo_simulation.md`](gazebo_simulation.md) | ROS 2/Gazebo setup, launch modes, mission lifecycle, topics, logging, troubleshooting |
| [`parameters.md`](parameters.md) | Scenario, controller, quadrotor, service-flight, reconfiguration, and Gazebo parameter reference |

## Validation hierarchy

```text
paper model
  double integrator / point mass
          |
          v
6-DoF quadrotor
  ideal thrust/torque realization
          |
          v
rotor-level quadrotor
  bounded allocation + motor dynamics
          |
          v
ROS 2 / Gazebo Harmonic
  Gazebo rigid-body, contact, and motor physics
```

The pure-Python models are selected with `--dynamics`. The Gazebo realization uses the same formation manager and high-level BLF/backstepping controller through the ROS 2 node `omrs_gazebo_controller`.
