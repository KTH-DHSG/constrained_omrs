# Pure-Python simulation

The pure-Python simulator is the fastest way to reproduce the open-team controller and inspect the effects of reconfiguration without starting ROS 2 or Gazebo.

## Installation

From the repository root:

```bash
uv venv --python /usr/bin/python3
uv pip install -e '.[simulation,test]'
```

`simulation` installs Matplotlib and Pillow for plots/GIF generation. `test` installs pytest.

## Basic run

The default dynamics are the **point-mass/double-integrator model considered in the paper**:

```bash
uv run --no-sync python examples/run_open_formation.py \
  --dimension 3 \
  --animation
```

Equivalent explicit command:

```bash
uv run --no-sync python examples/run_open_formation.py \
  --dimension 3 \
  --dynamics point_mass \
  --animation
```

## Dynamics modes

| `--dynamics` | Description | Dimensions |
|---|---|---|
| `point_mass` | Paper-level double integrator `p_dot = v`, `v_dot = u` | 2-D or 3-D |
| `quadrotor` | 6-DoF rigid body, geometric attitude/thrust realization, ideal commanded wrench | 3-D only |
| `quadrotor_rotor` | Same 6-DoF plant plus bounded rotor allocation and first-order rotor-thrust dynamics | 3-D only |

Examples:

```bash
uv run --no-sync python examples/run_open_formation.py --dimension 3 --dynamics quadrotor --animation
uv run --no-sync python examples/run_open_formation.py --dimension 3 --dynamics quadrotor_rotor --animation
```

The physical quadrotor modes use the same high-level controller with the quadrotor-specific CLI tuning and add a service-flight stage for a joining vehicle before prospective BLF control starts. The handoff is deliberately made before the service controller has completed the geometric rendezvous: prospective control owns the final approach, while capture checks keep the candidate away from the collision boundary and the relaxation initialization keeps the new edge comfortably inside its relaxed upper boundary.

## CLI reference

```text
--dimension {2,3}
    Spatial dimension. Default: 3.

--dynamics {point_mass,quadrotor,quadrotor_rotor}
    Plant model. Default: point_mass.

--output-dir PATH
    Output directory. By default:
    outputs/open_formation_<dimension>d_<dynamics>/

--animation
    Save GIF and, when supported by the local Matplotlib writer, MP4 animation.

--animation-fps N
    Animation frame rate. Default: 15.

--animation-stride N
    Use every N-th integration sample as an animation frame.
    Default is selected from the dynamics/dimension.

--tail-seconds T
    Length of 3-D trajectory tails. Default: 5 s.

--fixed-camera
    Disable the slow orbiting camera in 3-D animations.
```

Run `uv run --no-sync python examples/run_open_formation.py --help` for the authoritative CLI help.

## Default scenarios

### 3-D scenario

The main 3-D scenario contains seven robots. Robots `0--4` initially form the chain

```text
0 -- 1 -- 2 -- 3 -- 4
```

Robots `5` and `6` join at mission times `8 s` and `25 s`, each through two prospective edges. Robots `2`, `1`, and `3` request departure at `51 s`, `73 s`, and `95 s`, respectively. These departures exercise repeated make-before-break bridge creation.

The nominal distance interval is

```text
d_min = 0.45 m
d_max = 1.65 m
```

The simulation duration is `123 s` with `dt = 0.01 s`.

### 2-D scenario

The 2-D debugging scenario contains five robots, one join request and one removal. It is useful for faster controller/reconfiguration regression checks:

```bash
uv run --no-sync python examples/run_open_formation.py --dimension 2 --animation
```

## Outputs

`save_plots()` writes the standard diagnostic plots for the simulation. With `--animation`, the script additionally saves a GIF and attempts to save an MP4.

The script prints the final team size, final established edge set, topology-switch history, and, for quadrotor modes, acceleration-realization and actuator statistics.

## Programmatic API

The simulation can also be called directly:

```python
from constrained_omrs import (
    ControllerConfig,
    SmoothBacksteppingController,
    default_open_formation_scenario,
    simulate,
)

scenario = default_open_formation_scenario(3)
controller = SmoothBacksteppingController(
    num_robots=scenario.num_robots,
    dimension=scenario.dimension,
    collision_distance=scenario.collision_distance,
    sensing_distance=scenario.sensing_distance,
    config=ControllerConfig(
        k1=1.0,
        k2=0.8,
        k3=2.5,
        max_virtual_speed=0.8,
    ),
)

result = simulate(scenario, controller, dynamics="point_mass")
```

`SimulationResult` contains the state histories, high-level commands, graph/prospective-edge histories, formation errors, switch events, and, for the physical modes, attitude/thrust/torque and rotor histories.

## Changing the scenario

The default scenarios are defined in `src/constrained_omrs/scenario.py`. A custom `Scenario` specifies:

- desired positions (defined up to common translation for formation control),
- initial positions and velocities,
- initially active robots and graph edges,
- join and leave requests,
- `d_min` / `d_max`,
- reconfiguration settings,
- simulation duration and step size,
- optional 3-D charging-area configuration.

See [`parameters.md`](parameters.md) for the full parameter reference.

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest -v
```

The test suite covers BLF derivatives, saturation, reconfiguration/switching, simulation, quadrotor/rotor behavior, Gazebo adapters, and static ROS/Gazebo assets.
