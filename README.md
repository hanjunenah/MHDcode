# mhd2d

`mhd2d` is a compact C++17 finite-volume solver for dimensionless 2.5-D
resistive magnetohydrodynamics. Fields vary in `x` and `y`, while velocity,
magnetic field, and current retain three components. Python problem files define
the grid, initial state, runtime settings, and optional impressed-current
antennas.

## Features

- Conservative total-energy ideal-MHD update
- Limited MUSCL reconstruction, HLLD fluxes, and SSP-RK2 integration
- Hyperbolic GLM divergence cleaning
- Constant scalar resistivity with an explicit diffusion stability limit
- Periodic boundaries in both directions
- Multithreaded C++ worker pool
- Python-defined scalar, vector, and antenna profiles
- Compressed, append-only HDF5 output
- Interactive plotting for reconnection and antenna-driven problems
- Harris-sheet, Alfvén-antenna, and Orszag-Tang examples

## Requirements

- A C++17 compiler
- HDF5 development libraries
- `pkg-config`
- Python 3 with NumPy, h5py, and Matplotlib

On macOS with Homebrew:

```sh
brew install hdf5 pkg-config
python3 -m pip install numpy h5py matplotlib
make
```

The code uses `std::thread`; OpenMP is not required.

## Quick start

Build and run the short test suite:

```sh
make
make test
```

Run one of the supplied problems:

```sh
python3 run.py input/harris.py
python3 run.py input/alfven_antenna.py
python3 run.py input/orszag_tang.py
```

Existing output files are protected. Use `--force` to replace one, and use
command-line overrides for short exploratory runs:

```sh
python3 run.py input/harris.py --threads 4 --t-end 1.0 \
    --output output/harris_short.h5 --force
```

At each save interval, the solver reports wall time for the interval,
cumulative elapsed time, and estimated time remaining.

## Equations

The normalization uses `mu_0 = 1`. Without an antenna, pre-Maxwell Ampère's law
is

```text
J = curl(B).
```

The evolved conservative system is

```text
d rho/dt       + div(rho v) = 0,
d(rho v)/dt    + div[rho vv + (p + |B|^2/2) I - BB] = S_m,
d B/dt         + div[B tensor v - v tensor B + psi I - eta grad(B)] = S_B,
d E/dt         + div[(E + p + |B|^2/2) v - B(v dot B)] = S_E,
d psi/dt       + c_h^2 div(B) = -alpha psi,
```

with

```text
E = p/(gamma - 1) + rho |v|^2/2 + |B|^2/2.
```

To make the induction sign unambiguous, its ideal component is

```text
d B_i/dt + d/dx_j (v_j B_i - B_j v_i) = 0.
```

Pressure is recovered from total energy. In smooth ideal flow this is
equivalent to adiabatic pressure transport, while retaining conservative shock
heating.

### GLM cleaning

At the start of each step,

```text
c_h = glm_speed_factor * max_cells,directions(|v_n| + c_fast,n),
alpha = glm_damping * c_h / min(dx, dy).
```

The damping update is

```text
psi <- psi exp(-alpha dt).
```

`glm_speed_factor = 1` uses approximately the fastest physical signal speed.
Increasing it transports divergence errors faster but reduces the CFL time
step. Setting `glm_damping = 0` leaves the cleaning subsystem hyperbolic and
undamped.

### Resistivity

Constant `eta >= 0` adds componentwise magnetic diffusion,
`eta Laplacian(B)`. Face gradients are shared by adjacent finite volumes, and
the timestep includes the explicit diffusion limit.

### Impressed-current antennas

An antenna is separated from the plasma current through

```text
curl(B) = J_plasma + J_antenna.
```

Because the conservative Maxwell-stress flux already contains
`curl(B) cross B`, the added source terms are

```text
S_m = -J_antenna cross B,
S_B = eta curl(J_antenna),
S_E = -v dot (J_antenna cross B) - eta J_antenna dot J_plasma.
```

The energy source equals the external electrical work `-J_antenna dot E`, with
`E = -v cross B + eta J_plasma`. Antenna power and cumulative injected energy
are written to the output file.

## Python problem files

A problem defines `grid`, `solver`, and `output` dictionaries:

```python
grid = {"nx": 128, "ny": 128, "lx": 4.0, "ly": 4.0}

solver = {
    "gamma": 5.0 / 3.0,
    "resistivity": 0.0,
    "t_end": 2.0,
    "threads": 0,  # all logical cores; use 1 for serial execution
}

output = {"file": "output/results.h5", "interval": 0.2}
```

The fluid state is returned by a vectorized Python function. `rho` and `p` are
required; omitted velocity components default to zero.

```python
def initial_state(x, y):
    return {"rho": 1.0, "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.2}
```

Initialize the magnetic state with either

```python
def magnetic_field(x, y):
    return Bx, By, Bz
```

or

```python
def current_density(x, y):
    return Jx, Jy, Jz
```

If only current is supplied, the launcher reconstructs a periodic,
divergence-free magnetic field by inverting the discrete curl with FFTs.
`mean_magnetic_field` specifies the undetermined mean. If both functions are
present, the launcher checks their discrete Ampère compatibility.

### Antenna profiles

The antenna interface follows the same spatial-times-temporal pattern used by
SMILEI:

```python
def antenna_space(x, y):
    return amplitude * np.exp(-((x-x0)**2 + (y-y0)**2) / (2.0*width**2))

def antenna_time(t):
    return np.sin(omega*t)

antennas = [{
    "field": "Jz",  # Jx, Jy, or Jz
    "space_profile": antenna_space,
    "time_profile": antenna_time,
}]

antenna_sampling_interval = 0.01
```

Multiple antennas are summed. An inseparable function is also supported:

```python
antennas = [{
    "field": "Jy",
    "space_time_profile": lambda x, y, t: profile(x, y, t),
}]
```

`space_time_profile` cannot be combined with `space_profile` or
`time_profile`. Python profiles are sampled before the run and linearly
interpolated at both Runge-Kutta stages. Choose `antenna_sampling_interval` to
resolve the shortest imposed timescale.

## Supplied problems

| Problem | Input | Purpose |
|---|---|---|
| Double Harris sheet | [`input/harris.py`](input/harris.py) | Periodic resistive reconnection |
| Current antenna | [`input/alfven_antenna.py`](input/alfven_antenna.py) | Localized driven MHD-wave response |
| Orszag-Tang vortex | [`input/orszag_tang.py`](input/orszag_tang.py) | Shock interaction and turbulence benchmark |

### Orszag-Tang validation

The Orszag-Tang input uses the conventional periodic unit-square setup at
256 x 256 resolution:

```text
gamma = 5/3,  rho = 25/(36 pi),  p = 5/(12 pi),
vx = -sin(2 pi y),               vy = sin(2 pi x),
Bx = -sin(2 pi y)/sqrt(4 pi),    By = sin(4 pi x)/sqrt(4 pi).
```

The bundled
[`reference/orszag_tang_athena_256.npz`](reference/orszag_tang_athena_256.npz)
was generated with the official Athena++ code using 256 x 256 cells, HLLD,
second-order PLM/VL2, and constrained transport. Its exact source commit is
stored in the file.

Run the comparison at `t = 0.5`:

```sh
python3 run.py input/orszag_tang.py --force
python3 compare_orszag_tang.py output/orszag_tang.h5 \
    --save output/orszag_tang_comparison.png
```

For the supplied run, the density and pressure correlations with Athena++ are
`0.99953` and `0.99948`; their relative L2 differences are `0.93%` and `1.50%`.
The comparison tool also reproduces the standard pressure cuts at `y=0.4277`
and `y=0.3125`.

## Output and visualization

Each run writes one compressed HDF5 file:

```text
/x, /y, /time
/fields/rho, u, v, w, p, Bx, By, Bz, Jx, Jy, Jz, divB, psi
/fields/Jx_antenna, Jy_antenna, Jz_antenna
/diagnostics/mass, momentum_*, magnetic_flux_*, total_energy, ...
/diagnostics/antenna_power, antenna_energy_injected, max_abs_antenna_current
```

Field datasets have shape `(frames, ny, nx)`. `Jx`, `Jy`, and `Jz` are the
total Ampère current `curl(B)`; subtract the corresponding antenna field to
obtain the plasma current.

Open the Harris/reconnection viewer:

```sh
python3 plot_results.py output/harris.h5
```

Open the antenna-focused viewer:

```sh
python3 plot_antenna_results.py output/alfven_antenna.h5
```

Both viewers provide a time slider and arrow-key navigation. Render without a
GUI using `--frame` and `--save`:

```sh
MPLBACKEND=Agg python3 plot_results.py output/harris.h5 --frame -1 \
    --save output/harris_final.png
```

## Numerical limitations

- Boundaries are currently periodic only.
- Resistivity is constant, scalar, and explicit.
- The resistive induction update omits the local resistive Poynting-flux
  correction, so it is not intended for detailed local Ohmic-heating studies.
- GLM cleaning controls but does not eliminate discrete magnetic divergence.
  In the 256 x 256 Orszag-Tang test at `t=0.5`,
  `dx*max(abs(divB)) = 3.39e-2` at localized shock cells and
  `dx*mean(abs(divB)) = 3.51e-4`.
- The antenna time table is stored in memory; fine sampling of large grids can
  be memory intensive.
- This is a compact research and teaching code, not a production MHD package.

## Repository layout

```text
src/main.cpp                 C++ finite-volume solver
run.py                       Python input preparation and launcher
input/                       Example problem definitions
plot_results.py              Harris/reconnection viewer
plot_antenna_results.py      Antenna-focused viewer
compare_orszag_tang.py       Athena++ benchmark comparison
reference/                   Compact benchmark reference data
Makefile                     Build and smoke tests
```
