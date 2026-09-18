# Architecture

## System-level view

The implementation separates **open-team coordination** from the physical realization of the commanded translational acceleration.

```text
formation manager
    |
    | established/prospective graph + desired relative displacements
    v
BLF / backstepping controller
    |
    | desired translational acceleration u_i
    +-------------------------------+
    |                               |
    v                               v
point-mass dynamics            geometric quadrotor controller
                                    |
                                    | thrust + torque
                                    v
                              bounded rotor allocation
                                    |
                                    v
                         rotor dynamics / Gazebo motor model
```

The main implementation modules are:

| Module | Role |
|---|---|
| `barrier.py` | Edge BLF and compactly supported collision-only barrier |
| `controller.py` | Smooth virtual-velocity and second-order backstepping controller |
| `reconfiguration.py` | Formation manager, prospective edges, admission/removal switches |
| `scenario.py` | Scenario, membership requests, desired geometry, reconfiguration settings |
| `simulation.py` | Pure-Python simulation at three plant-fidelity levels |
| `quadrotor.py` | Geometric controller, rigid-body/rotor model, bounded allocation |
| `charging.py` | Service-flight controller for takeoff/rendezvous/return-to-charge |
| `plotting.py` | Pure-Python plots and animations |
| `constrained_omrs_ros/gazebo_controller.py` | ROS 2 supervisory/control node |
| `constrained_omrs_ros/gazebo_results.py` | Gazebo logging and post-processing |
| `constrained_omrs_ros/gazebo_markers.py` | Native Gazebo interaction-edge visualization |

## Interaction graph

The interaction topology is **prescribed by the formation manager**; proximity does not automatically create connectivity edges. Physical range instead constrains whether an edge can be established and maintained.

An established edge must remain inside

\[
d_{\min}<d_{ij}<d_{\max}.
\]

For a prospective edge, only the upper constraint is relaxed:

\[
d_{\min}<d_{ij}<d_{\max}+\rho_{ij}.
\]

The collision boundary is never relaxed.

## Prospective-edge relaxation

The implemented protected margin is

\[
s_{ij}=d_{\max}+\rho_{ij}-d_{ij}-\varepsilon_a.
\]

The nominal relaxation is a constant-rate contraction. The actual rate is projected so that the relaxed upper-distance constraint remains feasible:

\[
\dot\rho_{ij}=\max\{-c_\rho,\,\dot d_{ij}-\lambda_\rho s_{ij}\},
\qquad \rho_{ij}>0,
\]

with non-negativity enforced at `rho = 0`. Away from the feasibility correction, the relaxation therefore reaches zero in finite time.

## Joining sequence

The theoretical/paper-level point-mass simulation can immediately create a prospective interaction for a joining robot. The physical quadrotor and Gazebo realizations add a service-flight layer so the physical vehicle does not enter BLF control directly from a charging pad.

```text
charging
  -> takeoff
  -> rendezvous / capture
  -> candidate-only prospective control
  -> admission
  -> smooth symmetric handover
```

Before admission, a joining robot is not yet an active member of the OMRS. Its prospective formation edges act on the candidate only. Collision avoidance remains a physical pairwise safety mechanism.

In the Gazebo experiment, prospective control must persist for a minimum time and admission is subject to a relative-formation-error guard. After admission, the node blends the complete pre-switch candidate-only controller and the complete post-switch symmetric controller using a smooth quintic handover.

## Departure sequence

When a robot can leave without disconnecting the remaining graph, it is removed after the configured switch dwell time. If its removal would disconnect the graph, the formation manager uses make-before-break reconfiguration:

```text
departure request
  -> identify components of the surviving graph
  -> assign prospective bridge edge(s)
  -> recover nominal interaction range
  -> activate bridge edge(s)
  -> remove departing robot
  -> return to charging station (physical simulations)
```

The desired formation is modified only when required to obtain a feasible bridge, and retained desired relative displacements are preserved through rigid component translations whenever possible.

## Collision avoidance versus connectivity

Connectivity edges already contain the lower-distance BLF. Non-edge physical pairs inside the activation range receive a compact collision-only barrier. Its value, gradient, and Hessian vanish at the activation boundary, allowing the collision-neighbor set to change without a discontinuity in the virtual velocity or its derivative.

This separation is important for joining robots: an existing team member may repel a nearby candidate for collision safety without being attracted toward the candidate's prospective formation target.

## Physical realization

For `quadrotor` and `quadrotor_rotor`, the high-level acceleration command is mapped to a desired attitude and collective thrust by `GeometricQuadrotorController`. `quadrotor_rotor` additionally solves a bounded four-rotor allocation problem and propagates first-order rotor-thrust dynamics.

The Gazebo path uses the same high-level logic but Gazebo owns the rigid-body, contact, and motor dynamics. The ROS node publishes rotor angular-speed commands and receives odometry through `ros_gz_bridge`.
