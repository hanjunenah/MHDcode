# mhd2d

`mhd2d` is a compact C++17 finite-volume solver for dimensionless 2.5-D
resistive magnetohydrodynamics. Fields vary in `x` and `y`, while velocity,
magnetic field, and current retain three components. Python problem files define
the grid, initial state, runtime settings, and optional current antennas.

Thruster input files (`input/*thruster*.py`) and all generated `output/`
files are local-only and excluded from the GitHub repository. The published
checkout includes the generic example inputs below. Optional local-study
notes refer to separately retained files and results.

## Features

- Conservative total-energy ideal-MHD update
- Limited MUSCL reconstruction, HLLD fluxes, and SSP-RK2 integration
- Hyperbolic GLM divergence cleaning
- Constant scalar resistivity with an explicit diffusion stability limit
- Periodic, zero-gradient outflow, or absorbing boundaries in each direction
- Optional stationary impenetrable slip bodies with resistive magnetic penetration
- Powered planar current-density constraints independent of plasma resistivity
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

On this workspace's Homebrew installation, `pkg-config hdf5` selects an MPI
build whose flags omit the MPI include directory. To build with the installed
serial HDF5 instead, use:

```sh
PKG_CONFIG_PATH=/opt/homebrew/opt/hdf5@1.14/lib/pkgconfig make
```

## Optional local thruster studies

This section describes local inputs and outputs that are not distributed
with the source. Its commands require those separately retained files.

The local fiducial thruster is the standalone
`input/reconnection_thruster_injection_open.py`.
It combines the former localized-plasma, injection and open-exterior input
files into one case. Its defaults are:

- 256 x 128 cells on a 36 x 24 domain, with open-front powered plates.
- Background-only initial fluid density 0.01 and uniform initial pressure 0.01;
  no dense initial propellant load. Solid states are fixed placeholders.
- Continuous plasma feeding near the throat at 0.01 code mass/time
  (approximately 0.0833 mg/s with the example SI normalization), zero feed
  velocity and specific internal energy 0.015.
- Plasma resistivity 0.001, angular frequency 0.05, peak current scale about
  0.751282, ten drive cycles and eight worker threads.
- Extrapolated exterior face states with a two-unit weak sponge: fluid
  relaxation 0.02 and extra magnetic diffusivity 0.001.

The inlet profile is normalized on the mesh to keep the specified total
feed rate independent of spatial sampling. The finite background keeps the
single-fluid MHD equations well-defined; it is not vacuum. Uniform initial
pressure avoids an initial gas-pressure launch, so the tenuous background is
hotter than the fed plasma. Results are written under
`output/thruster/injection/open/`.

`MHD_THRUSTER_MASS_FLOW_CODE` sets the feeding rate. Set
`MHD_THRUSTER_ABSORPTION_RATE`, `MHD_THRUSTER_BUFFER_ETA`,
`MHD_THRUSTER_BOUNDARY_EXTERIOR`, or `MHD_THRUSTER_BUFFER` to tune the boundary
layer. This is an extrapolated-face/weak-sponge approximation, not an exact
nonreflecting or vacuum boundary.

Eight workers had the best repeated 256 x 128 benchmark median: 4.06 s,
versus 4.12 s with ten and 4.29 s with fourteen workers for the same 508 steps.
Eight and ten are effectively tied; eight uses fewer workers. The benchmark
script and summary are under
`output/thruster/injection/validation/thread_benchmark/`.
`MHD_THRUSTER_THREADS` and `--threads` override the choice.

For a closed-front comparison with the fed fiducial case, use:

```sh
python3 run.py input/reconnection_thruster_closed_front.py
```

This standalone input matches the open-front fiducial's 256 x 128 grid,
36 x 24 domain, background-only initial plasma, continuous feed, drive,
duration, weak outflow buffer, SI scaling and eight workers. A
constant-thickness circular nose
joins the two plate centerlines tangentially and closes the front opening.
The current profile continues around the nose, with the same peak current
density and only downstream tip tapering. The downstream arms are unchanged.
This is a powered U-shaped antenna: at the default 256 x 128 sampling its
integrated current is about 14% larger, so this compares antenna shapes at
fixed peak current density, not fixed total current. Thermal annotations are
recomputed for the new geometry without renormalizing the drive. New results
are written separately under
`output/thruster/injection/closed_front/`. This fed closed-front case has
completed ten cycles in 875.7 s (14 min 35.7 s), taking 102,430 steps and
saving 321 frames through `t=1256.6370614`.

The local uniform-initial-plasma reference is
`input/reconnection_thruster.py`. It uses a stronger ambient/sponge boundary
and defaults to 512 x 256 with ten workers. Historical uniform-plasma
closed-front results were removed during cleanup; they did not represent
the new fed comparison.

Earlier localized-initial-load results were also removed. Strong-buffer fed
comparison data, archived inputs and validation reports remain locally under
`output/thruster/injection/`. The separate localized and
injection input files have been consolidated into the fiducial file above.

Retained local thruster results live under `output/thruster/`, which now
contains only `injection/` and `fiducial/`. The local fiducial input is
`input/reconnection_thruster_injection_open.py`, writing to
`output/thruster/injection/open/`. It drives the actual current density
inside stationary nonplasma plates, using `solver["antenna_model"] =
"prescribed_current"`. The plates admit magnetic field; their current is
independent of plasma resistivity. The plasma and absorbing outer boundaries
evolve self-consistently. The historical resistive-impressed fiducial runs remain under
`output/thruster/fiducial/`. Retired conductor runs, earlier resistive trials,
old uniform-plasma outputs and root-level validation/domain/design archives
were removed during cleanup. Retained fed-case validation and boundary
comparisons are inside `output/thruster/injection/`.

The current open-front fed fiducial completed ten cycles on 256 x 128 in
830.8 s (13 min 50.8 s), taking 97,632 steps and saving 321 frames through
`t=1256.6370614`. Its result is under
`output/thruster/injection/open/eta_0.001/`. Validation passed: minimum
saved density and gas pressure are 0.007816 and 0.007475, no saved plasma
cell reaches either floor, and relative mass/energy budget residuals are
3.34e-13 and 6.88e-13. The full report is
[`result_validation.json`](output/thruster/injection/open/validation/production_20261118/result_validation.json).

The fed closed-front run also passed validation. Its saved fields are
finite, body velocity remains zero, and no saved plasma cell reaches a
density or pressure floor. Minimum density and gas pressure are 0.005521
and 0.003233; relative mass/energy residuals are 8.10e-13 and 6.71e-13.
Its full report is
[`result_validation.json`](output/thruster/injection/closed_front/validation/production_20261118/result_validation.json).

The final five drive cycles give the following body-thrust and apparent
feed-based specific-impulse diagnostics with the example SI scaling:

| Fed configuration | Mean useful thrust (mN) | Cycle-thrust standard deviation (mN) | Apparent feed-based Isp (s) |
| --- | ---: | ---: | ---: |
| Open-front fiducial | 0.35174 | 0.00358 | 430.56 |
| Closed front | 0.19810 | 0.01262 | 242.49 |

This comparison holds peak current density fixed; the closed front carries
14.2% more integrated absolute current. Feeding, drive and boundary settings
match. The selected peak current is 0.112% above the estimated closed-front
thermal ceiling; the hardware annotations retain this difference instead of
renormalizing the comparison. These diagnostics retain the background-plasma, boundary and
resolution limitations described below. The comparison data and plot are
[`fed_front_comparison.json`](output/thruster/injection/comparison/fed_front_comparison.json),
[`fed_front_cycles.csv`](output/thruster/injection/comparison/fed_front_cycles.csv)
and [`fed_front_thrust_Isp.png`](output/thruster/injection/comparison/fed_front_thrust_Isp.png).

The previous prescribed-current production run at current amplitude 1 and
omega=0.005 completed three cycles on 256 x 128 at eta=0.001 in 102,336 steps
(14 min 38 s), with 97 saved frames.
Its input snapshot and result artifacts were removed during cleanup.
Saved density and pressure stayed positive, the plate current was enforced to
1.9e-14, and startup matches the retained quarter-cycle reference exactly.
The compact buffer is strongly contacted (peak entrance density perturbation
38.8%), and cycle-to-cycle fields are still evolving; these results do not
establish converged thrust or efficiency.

The fiducial thruster defaults to the padded domain: 36 x 24 at 256 x 128,
with a two-unit weak sponge. The plate front is at x=9 and the throat center at y=12.
Plate length, slope, gap, thickness, and current profile retain their physical
sizes; relative to the compact box, the thruster translates by (+3,+6)
without rescaling. The fed closed-front comparison uses the same grid and
box as the fiducial case. The uniform-plasma reference defaults to 512 x 256
on the same domain.

Defaults include plasma resistivity `eta=0.001`, angular frequency
`omega=0.05`, and a passive-copper thermal current ceiling (nominal code
amplitude about 0.751282). Ten drive periods end at `t=1256.637061`.
The fiducial defaults to eight worker threads based on repeated timings.
The uniform-plasma reference retains ten workers from its earlier benchmark.
Set `MHD_THRUSTER_THREADS` or `--threads` to override.
Set `MHD_THRUSTER_ETA`, `MHD_THRUSTER_CURRENT_SCALE`, `MHD_THRUSTER_OMEGA`, `MHD_THRUSTER_CYCLES`,
or `MHD_THRUSTER_NX`/`MHD_THRUSTER_NY` for physical/grid comparisons.
`MHD_THRUSTER_LX`, `MHD_THRUSTER_LY`, and `MHD_THRUSTER_BUFFER` override the
box and absorbing-layer width. Output filenames include box dimensions,
current amplitude and angular frequency, protecting older run products.
Set both `MHD_THRUSTER_CURRENT_SCALE=1` and `MHD_THRUSTER_OMEGA=0.005` to
reproduce the old drive parameters.

The default `MHD_THRUSTER_DOMAIN=padded` places the buffer 8.05 units from the
outer plate tips sideways, 7 upstream, and 18.33 downstream. A normal launch
uses this larger domain:

```sh
python3 run.py input/reconnection_thruster_injection_open.py
```

The compact preview remains available explicitly:

```sh
MHD_THRUSTER_DOMAIN=compact python3 run.py input/reconnection_thruster_injection_open.py
```

That uses 256 x 128 on 24 x 12, with clearances of only 2.05 sideways,
4 upstream, and 9.33 downstream. The previous amplitude-1, omega=0.005
compact quarter-cycle run already had about 0.65% density perturbation at
the side-buffer entrance. With the fiducial's fixed 256 x 128 resolution,
the compact option changes cell spacing as well as the box size. Change
resolution with box size to isolate boundary effects from plate resolution.
Even the padded domain encounters outgoing
waves within a drive cycle; compare near-thruster fields and cycle impulse
across domains before interpreting thrust or efficiency quantitatively.

The historical matched-spacing startup-quarter comparison (256 x 128 compact versus
384 x 256 padded) changes useful axial impulse by +2.21% and native circuit
work by +0.0507%; near-thruster momentum density differs by 3.88% in relative
L2 norm. A lower-resolution full-cycle comparison changes net impulse
magnitude by 69%, indicating substantial late boundary sensitivity; those
coarse plates remain underresolved for quantitative power or efficiency.
Neither comparison establishes domain-converged cycle thrust. The original
domain-comparison files were removed during cleanup; these values summarize
the historical measurements.

The historical resistive-impressed one-cycle fiducial reference reaches one full period in 34,112 steps and
3 min 2 s on this Mac. Its preview, frozen input, manifest, and validation
report are in `output/thruster/fiducial/eta_0.001/`. All saved states are
finite and positive, and mass/energy budgets close within 4.6e-12.
Late transverse flow reaches the side buffers; this run does not establish
domain independence. These timings and validation results describe the previous
drive model, not the prescribed-current update. Set `MHD_THRUSTER_CYCLES=1`
for a one-cycle run; the current fiducial runs ten cycles by default.

The historical resistive-impressed three-cycle run reaches `t=3769.911` in 102,336 steps and
9 min 13 s, with 97 saved frames. Its result has the suffix `_cycles3.h5`;
`validation_cycles3.json`, `cycle_evolution.png`, and
`fiducial_cycles3_preview.png` summarize it. All states remain finite and
positive and energy/mass residuals stay below 5.9e-12. Cycle-two-to-three
same-phase velocity fields still differ by roughly 13--19% in the core;
a periodic steady state has not been demonstrated. Side-buffer contact
also persists at late times.

At fixed cell count, shrinking the box decreases the diffusion-limited
timestep and increases the number of steps. Refine the fiducial grid with:

```sh
MHD_THRUSTER_NX=512 MHD_THRUSTER_NY=256 python3 run.py input/reconnection_thruster_injection_open.py
```

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
the timestep includes the explicit diffusion limit. This describes the default
`resistive_impressed` model. In `prescribed_current` mode, physical resistivity
acts only in the plasma induction equation; the powered plate current is a
separate constraint.

### Resistive impressed-current antennas

This is the backward-compatible default,
`solver["antenna_model"] = "resistive_impressed"`.

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
are written to the output file. This closure treats the prescribed antenna
profile as an offset in Ohm's law. It does not constrain the actual total
current inside a solid: from zero field and zero velocity, setting `eta=0`
removes its magnetic drive.

### Powered prescribed-current plates

`prescribed_current` is the default antenna option for the local thruster
and powered-plate inputs. Set `solver["antenna_model"] = "prescribed_current"`
to specify actual `Jz` inside stationary nonplasma bodies. This planar branch
requires nonperiodic boundaries, `w=Bz=0`, spatially uniform initial `Bx` and `By`, no `Jx` or `Jy`,
and current support confined to `solid_mask`. Other examples continue to use
the default 2.5-D impressed-current model.

The powered branch evolves a nodal vector potential with
`Bx = dAz/dy`, `By = -dAz/dx`. At powered nodes it enforces the discrete
Ampere constraint `-Laplacian(Az) = Jz`; plasma nodes evolve
`dAz/dt = -Ez` with the plasma's ideal and resistive electric field. Solving
the differentiated constraint supplies the plate electric field and a shared
interface EMF. The source supplies current and electrical work independently
of plasma `eta`. Magnetic flux can penetrate the plates. In an ideal plasma,
induced currents can screen the exterior field even when the powered current
and interior plate field are nonzero.

Cell-centered current samples are distributed conservatively to their four
corner nodes. All corners of solid cells, including zero-current surface
nodes, belong to the powered constraint. This preserves the integrated
prescribed current and prevents an unconstrained surface current from
canceling it inside the powered region. Its electromagnetic boundary extends
half a cell beyond the mechanical cell mask; refine the grid to assess this
geometry approximation. Current accuracy must be checked with the native
nodal Ampere diagnostic, since a centered curl of averaged cell fields has a
different stencil.

The prescribed profile represents a distributed ideal current supply; material
conductivity and the external feed circuit are not resolved. Native electrical
work is `integral(Jz * dAz/dt dV) = -integral(J dot E dV)` over powered nodes.
Face-field magnetic energy and the magnetic energy of averaged cell fields
differ at finite resolution. The source diagnostics distinguish this magnetic
averaging difference and discrete energy corrections from native circuit work:
`native_magnetic_energy` uses the face fields, and `magnetic_averaging_energy`
is native minus cell magnetic energy. `current_drive_energy_correction` is
`solid_constraint_energy - antenna_energy_injected`; it exposes the difference
between the discrete powered-body energy exchange and integrated native
circuit work. This is a source/body spatial coupling error, not plate heating
or a source to add to the plasma. It is distinct from the magnetic averaging
energy: matching a cell-centered body surface and Poynting flux to nodal current
and face fields also contributes to the error.

An independent static check solves the canonical powered Ampere constraint
with zero plasma potential and sets `dAz/dt = Az` at unit scaling rate. Native
circuit power then equals twice the native magnetic energy. Comparing that power with the
finite-volume body's magnetic-energy rate plus outgoing Poynting flux gives:

| Grid | FV body power / native circuit power | Relative error |
| --- | ---: | ---: |
| 96 x 48 | 0.648560 | -35.14% |
| 256 x 128 | 0.974545 | -2.55% |
| 512 x 256 | 0.997515 | -0.248% |

The 96 x 48 plates are underresolved for power or efficiency estimates. This
check demonstrates spatial convergence of the coupling; it does not replace
time-dependent resolution and domain checks for thrust. Machine-precision
closure of the budget including the correction is an accounting identity,
not evidence that physical circuit power is resolved. The auxiliary source
and result files for this historical static check were removed during cleanup.

## SI scaling, passive copper and drive frequency

The optional local thruster inputs use an explicit illustrative normalization:
`L0=0.01 m`, `B0=0.02 T`, singly ionized xenon `n0=1e18 m^-3`, and active
out-of-plane length `Dz=0.1 m`. Thus `J0=B0/(mu0*L0)=1.59155 A/mm²` and
`I0=B0*L0/mu0=159.155 A`. These references are assumptions, not SI units
deducible from a dimensionless run. The axial plate length is 6.67 cm,
thickness 4 mm, and throat gap 1 cm. At code amplitude 1, each plate carries
about 203 A peak (or equivalent ampere-turns for a winding pack).

For the requested passive cooling in vacuum, the default uses bare copper
with emissivity 0.05, a maximum copper temperature of 80°C, and a 300 K
radiative environment. Only the two outward broad faces radiate; there is no
convection, support-conduction or mutually facing inner-surface cooling credit.
The current ceiling solves

```text
P_rad = epsilon*sigma*A_outer*(T_max^4 - T_env^4)
P_Cu  = rho_Cu(T_max)*Dz*L0^2*J0^2*a^2*integral(g^2 dx*dy*)/2
P_Cu + P_external <= P_rad.
```

The exact continuum profile integral makes the selected nominal amplitude
independent of grid resolution. With zero external heating allowance,
`P_rad=0.286683 W` for the pair and `a=0.751282`, giving nominal peak density
1.19570 A/mm² and about **152.5 A peak / 107.8 A RMS per plate** on the default
grid. This is a conditional upper ceiling, not a guaranteed hardware maximum:
plasma/sheath heating, feeds and the actual mount are not modeled. Bare copper
has low and surface-dependent emissivity ([NASA material table](https://ntrs.nasa.gov/api/citations/20150021314/downloads/20150021314.pdf)).
Change `MHD_THRUSTER_EMISSIVITY`, `MHD_THRUSTER_COPPER_MAX_C`,
`MHD_THRUSTER_ENVIRONMENT_K`, and `MHD_THRUSTER_PLATE_HEAT_W` for an actual
thermal design. If external heat exhausts the radiation budget, preparation
fails. An explicit current-scale override is recorded as outside the selected
budget when appropriate. Copper resistivity is used only for this external
thermal estimate; plasma `eta=0.001` and the prescribed-current solver are
unchanged. No perfect-conductor or uniform-A condition is introduced.

Sweet–Parker scaling uses local reversing field and sheet length:
`VA=B_up/sqrt(rho_up)`, `S=L*VA/eta`, and
`tau_SP=L^(3/2)/sqrt(eta*VA)`. The value `eta^(-1/2)=31.6` additionally
assumes `L=VA=1`. The previous amplitude-1 run gives conditional estimates
141–217 using sampled upstream fields and a current-length proxy; the layer
is not a demonstrated resolved Sweet–Parker sheet. If field scales with
amplitude while density/length stay fixed, `tau_SP` scales as `a^(-1/2)`.
At the new current this suggests roughly 163–250, so `omega=0.0004` gives
`omega*tau_SP` around 0.065–0.100. This is a deliberately slow design choice,
not automatic feedback or a measured new reconnection rate. The ordinary
frequency is `f=omega/(2*pi)`, and the signed half-cycle is `pi/omega`.
[Sweet–Parker derivation](https://farside.ph.utexas.edu/teaching/plasma/Plasma/node101.html).

With the illustrative xenon density, the historical `omega=0.0004` drive is
about 243 Hz; copper skin depth at 80°C is about 4.7 mm. The predicted SP half-thickness is still
only about 1–2 default cells, so a frequency change does not establish
resolved reconnection. Also the ion inertial length at this SI density is
about 2.6 m, exceeding the device size: single-fluid resistive MHD does not
establish real Hall-size reconnection physics. These SI and thermal estimates
were originally documented in design files that were removed during cleanup.

The launcher preserves optional `physical_scaling` and `antenna_hardware`
attribute groups in completed HDF5 results; fields remain dimensionless.
Override reference values with `MHD_THRUSTER_L0_M`, `MHD_THRUSTER_B0_T`,
`MHD_THRUSTER_N0_M3`, and `MHD_THRUSTER_DEPTH_M`. Current integrals are through
the x-y cross-section and do not acquire an extra depth factor; depth enters
power and thermal area.

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
FFT initialization from current alone requires both axes to be periodic.
Supply `magnetic_field(x, y)` when using nonperiodic boundaries.

### Plasma injection

An optional `plasma_injection` dictionary adds plasma in a spatial region:

```python
plasma_injection = {
    "mass_rate": lambda x, y: 0.01*np.exp(-((x-2)/0.3)**2-((y-1)/0.2)**2),
    "u": 0.0, "v": 0.0, "w": 0.0,
    "specific_internal_energy": 0.015,
    "time_profile": 1.0,
}
```

`mass_rate` is density added per unit time and volume, before multiplication
by `time_profile`. Spatial values may be scalars or functions of `(x, y)`.
Feed velocities default to zero; `specific_internal_energy` is required and
must be positive wherever mass is added. For injected pressure and density,
use `specific_internal_energy = p_inj / ((gamma-1)*rho_inj)`.
The solver adds the conservative sources

```text
S_rho = q
S_momentum = q * v_inj
S_energy = q * (specific_internal_energy + |v_inj|^2/2)
q = mass_rate * time_profile(t)
```

This is mass addition and mixing, not a density reset. It supplies no magnetic
field and does not remove existing plasma. A callable `time_profile(t)` also
requires a positive `sampling_interval` in this dictionary. Values are
linearly interpolated, with timesteps ending at temporal knots; rates and
temporal amplitudes must be finite and nonnegative. Injection must be zero
inside solids. The planar `prescribed_current` model requires injected `w=0`.
Omit the dictionary, or set it to `None`, to disable injection.

Result files retain the sampled profiles under `/plasma_injection` and add
cumulative `injection_mass`, `injection_momentum_x/y/z` and `injection_energy`
diagnostics. Add `injection_mass` to the plasma mass budget and
`injection_energy` to the global energy budget; both exclude the initial
load. The antenna viewer includes this feed energy in its plotted balance.
Nonzero feed velocity supplies external momentum; feed-hardware reaction is
outside the antenna's `solid_force_*` diagnostic. Very strong feeding can
change wave speeds within a timestep and requires a smaller CFL and checks
of temporal resolution. This single-fluid source represents already ionized
plasma; neutral transport and ionization energy are not modeled.

### Outer boundaries

Set `solver["boundary_x"]` and `solver["boundary_y"]` to `"periodic"` (the
default), `"outflow"`, or `"absorbing"`. Outflow extrapolates the edge state with zero normal
gradient. It removes periodic wraparound, permits backflow, and is not a
characteristic nonreflecting condition. Keep the boundaries well away from
the driven region and compare larger domains before interpreting late flow.
The boundary choice is saved in the output and used by both plotting scripts.

Absorbing boundaries combine an ambient exterior Riemann state with a smooth
buffer inside the domain. For example:

```python
solver = {
    "boundary_x": "absorbing",
    "boundary_y": "absorbing",
    "absorption_width": 3.0,                 # physical length from each face
    "absorption_rate": 0.5,                  # maximum fluid relaxation rate
    "absorption_magnetic_diffusivity": 0.05, # additional buffer diffusivity
    "absorption_exterior": "ambient",        # ambient or outflow
}
boundary_ambient = {"rho": 1.0, "p": 0.01}  # u/v/w/Bx/By/Bz default to zero
```

The rate and extra diffusivity rise quadratically from zero at the inner edge
of the buffer. Density, momentum, and internal energy relax toward the explicit
ambient state. Magnetic damping uses `-curl(eta_buffer curl(B))`, preserving the
solver's centered discrete divergence; it does not directly relax individual
magnetic components toward the ambient field. Its magnetic energy change goes
to the artificial absorber reservoir while local thermal pressure is preserved.
With `absorption_exterior="ambient"` (the backward-compatible default), the
exterior Riemann state preserves the interior normal magnetic field and GLM
field and uses ambient tangential fields and fluid variables. Super-fast
outward flow uses the interior state. With `absorption_exterior="outflow"`,
the face state is extrapolated from the interior even for sub-fast flow;
the volume sponge still operates independently. This permits the weak-sponge
case to pass the outgoing plume at its outer faces without replacing that
face state by a stationary reservoir. It also permits backflow, as ordinary
zero-gradient outflow does. The physical `resistivity` in the analysis region
is unchanged.

This is a sponge/far-field approximation, not an exact nonreflecting condition
or perfectly matched layer. Absorption depends on wavelength, incidence angle,
layer width, and damping strength; excessive relaxation can itself reflect
magnetic waves. Resolve the buffer with several cells, keep antennas and the
measurement region outside it, and check domain/width sensitivity. The width
must be below half each absorbing domain dimension and contain an active cell
center. Overlap with stationary bodies is rejected. Both viewers mark the
inner buffer boundary with dotted lines.

The ambient reservoir can supply or remove mass, momentum, and energy.
`absorption_mass_exchange`, `absorption_momentum_exchange_x/y/z`, and
`absorption_energy_exchange` are cumulative signed exchanges, positive when
supplied to the computational domain. They are artificial boundary effects,
not antenna thrust or physical heating. The energy and plasma-mass balances
are, for the default impressed-current model:

```text
E(t) - E(0) = antenna_energy_injected + solid_constraint_energy
            + absorption_energy_exchange - boundary_energy_outflow
M_plasma(t) - M_plasma(0) = absorption_mass_exchange - boundary_mass_outflow.
```

For prescribed-current plates, the energy balance is instead

```text
E(t) - E(0) = antenna_energy_injected + current_drive_energy_correction
            + absorption_energy_exchange - boundary_energy_outflow.
```

The first two terms together equal `solid_constraint_energy` in this mode;
do not add native circuit work to that constraint exchange a second time.

The uniform-plasma reference, `input/reconnection_thruster.py`, uses this mode on a 512 x 256 grid,
with plasma eta=0.001, stationary powered plates, and continuous drive.
Its 36 x 24 domain has a two-unit buffer on every side, with magnetic
buffer diffusivity 0.05 in addition to the physical plasma resistivity.
The fiducial `input/reconnection_thruster_injection_open.py` uses the gentler
extrapolated faces at 256 x 128. Boundary sensitivity experiments
and outgoing fast-wave tests are saved under
`output/thruster/injection/boundary_comparison/`.
In a controlled ten-cycle 128 x 96 comparison, the last five cycles' net
sponge mass removal fell from 91.1% to 18.1% of the injected mass, while net
outer-face outflow rose from 1.97% to 87.5%. These are mixture mass balances
normalized by feeding, not traced propellant fractions; changing domain
inventory can make their sum differ from 100%. Mean body thrust changed by
1.39% on that grid. A separate small-amplitude outgoing fast-wave test against
an extended-domain reference reduced returned fast-mode amplitude from
1.26% to 0.378%. This demonstrates improvement for those tests, not general
nonreflection or resolution-converged thrust. The source/body geometry in
the coarse comparison is underresolved. These boundary comparisons used
128 x 96. The ten-cycle 256 x 128 weak-boundary open-front fiducial has now
completed and passed its numerical checks. This alone does not establish
boundary or resolution convergence.

### Fixed current-carrying bodies

An antenna profile already stays fixed in space. By itself it does
not exclude plasma from the antenna region. To add a stationary material
region, define a vectorized `solid_mask(x, y)` returning booleans or 0/1:

```python
def solid_mask(x, y):
    return (x > 1.0) & (x < 1.4) & (y > 2.0) & (y < 4.0)
```

The solver enforces exactly zero mass flux at plasma/body faces and keeps
body velocity zero. The wall is free slip because this solver has no viscosity.
In the default `resistive_impressed` model, magnetic field and GLM cleaning
evolve through the body using the same uniform resistivity as the plasma.
In `prescribed_current` mode, the powered Ampere constraint determines the
body field and the shared Faraday EMF couples it to the plasma. Body density
and thermal pressure are fixed numerical placeholders in both modes, and
initial body velocity must be zero. Neither mode resolves material
conductivity, antenna temperature, or structural deformation.

The continuous thruster confines the prescribed current to the bodies, uses
smooth current tapers, and drives a sinusoid throughout the run after a smooth
startup. Its 36 x 24 box uses absorbing boundaries. It does not subtract a diffuse periodic return
current; the external circuit lies outside the computational domain.

Body interfaces follow cell faces, so oblique plates have a staircase boundary
and a first-order electromagnetic interface with numerical diffusion. Refine
the grid to assess sensitivity. This is a rigid boundary approximation, not
a viscous no-slip, thermal, or structural simulation.

`/solid/mask` is saved once in the HDF5 output. Both viewers outline bodies and
hide their placeholder fluid states. The antenna viewer reconstructs magnetic
contours without periodic extension for open axes and includes boundary/body
terms in its energy balance. Open-domain contour reconstruction is a
least-squares solenoidal projection when magnetic divergence is nonzero.

Global `total_energy` includes body placeholders and magnetic energy. Prefer
`plasma_mass`, `plasma_energy` and its magnetic/kinetic/thermal components for
plasma diagnostics. Additional diagnostics include `solid_magnetic_energy`,
`solid_constraint_energy`, `boundary_energy_outflow`, `boundary_mass_outflow`,
`solid_force_*`, `solid_impulse_*`, and `max_wall_mass_flux`. Positive
`solid_force_*` is traction on the body; the support reaction has opposite sign.
In the default impressed-current model, energy accounting without an absorbing
layer obeys:

```text
E(t) - E(0) = antenna_energy_injected
            + solid_constraint_energy - boundary_energy_outflow.
```

The constraint reservoir holds the body pressure and velocity fixed; it must
not be interpreted as an independently resolved antenna temperature or heat
load. Current/body force, external electrical work and plasma energy are
different diagnostics. For prescribed-current plates, use the powered energy
balance above, omitting `absorption_energy_exchange` when no buffer is present.

Run the boundary/body and plotting verification without changing example
outputs:

```sh
python3 -m unittest discover -s tests -v
```

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

Cell-field datasets have shape `(frames, ny, nx)`. `Jx`, `Jy`, and `Jz` are the
centered curl of cell magnetic fields. In the default impressed-current mode,
subtract the corresponding antenna field to obtain the plasma current.
For prescribed-current plates, use the additional native nodal Ampere fields
and their powered mask to assess enforcement of the current constraint:

```text
/prescribed_current/x, y, powered_mask
/prescribed_current/Az, Jz, Jz_target
/diagnostics/max_current_constraint_error, max_stage_current_constraint_error
/diagnostics/max_abs_ct_divB, native_magnetic_energy, magnetic_averaging_energy
/diagnostics/current_drive_energy_correction
```

The native fields have shape `(frames, ny+1, nx+1)`, with a static nodal
`powered_mask`; `Jz_target` is the conservative nodal projection of the supplied
cell current. Constraint errors compare native `Jz` with that target only on
powered nodes. In this mode `/fields/divB` and `max_abs_ct_divB` measure the
native finite-volume divergence of the face fields; the output attribute
`divB_definition` records this definition. A centered curl or divergence of
averaged cell fields uses a different stencil. The native circuit
power and its time integral use the existing `antenna_power` and
`antenna_energy_injected` diagnostics. In this mode `antenna_power` is the
last integration step's RK2 average, rather than an instantaneous output-time
sample. The antenna viewer plots native circuit work and the discrete drive
correction separately and includes the current-constraint error in its title.
When saved nodal `Az` is available, the antenna viewer averages it to cell
centers for magnetic contours and recomputes native CT divergence from its
face fields, avoiding a false residual from the centered cell-field stencil.

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

The antenna viewer shows momentum density `rho*v` in the selected velocity
direction. Its density-perturbation scale uses the 98th percentile of
`abs(rho-rho_initial)` over all plasma frames so narrow density peaks do not
hide weaker disturbances. The scale stays fixed while moving the slider;
colorbar extensions mark saturated values. Set `--density-limit 0.01` to use
an explicit symmetric range of -0.01 to +0.01.

For separately retained local thruster data, open the dedicated viewer.
This example uses the local completed ten-cycle 256 x 128 open-front
fiducial; its HDF5 file is not included in the GitHub checkout:

```sh
python3 plot_thruster_results.py output/thruster/injection/open/eta_0.001/thruster_injection_openbc_256x128_L36x24_rho_bg0.01_mdot0.01_J0.75128226_omega0.05_cycles10_rate0.02_etaBuffer0.001_outflow.h5
```

It defaults to the final frame, axial velocity and momentum, and the whole
simulation domain. Orange dashed outlines mark the plasma source on the spatial
panels. The local injection map, source peak, integrated feed rate and
cumulative added mass follow the time slider. The viewer reads these from
the result's archived `/plasma_injection` profiles and diagnostics, so later
input edits do not change the displayed source. It shows the actual axial
velocity `v_x`, absolute gas pressure `p`, and absolute density on a logarithmic
scale. The center spatial panel shows the plasma electric field
`E_z = v_y B_x - v_x B_y + eta J_z` reconstructed from Ohm's law for
prescribed-current runs, with the plates masked and a signed color range
fixed across all frames. In the legacy resistive-impressed mode, the
resistive term uses `J_z - J_z_antenna`.

The bottom-center panel combines thrust on the left axis with apparent
feed-based specific impulse on the right axis. The specific-impulse curves
use stored body impulse divided by mass injected over matching output
intervals (dotted) and cumulatively from the first frame (dashed). Values
remain in seconds even with `--units code`; SI reference metadata is required
for this conversion. Zero-feed windows and missing reference scales are
marked undefined. The slider moves its time marker along with the thrust and
feed histories. This is a signed quantity, positive for useful thrust toward
`-x`, and it includes the background-plasma limitations described below.

The thrust panel shows the saved last-step force, output-interval means from
stored impulse, and completed-cycle means when frequency metadata is present.
Positive thrust is force on the antenna toward `-x`, opposite the `+x` opening.
Cycle boundaries between saved frames use interpolated impulse and are labeled
accordingly. SI time, force and feed units use the saved reference scales,
including the out-of-plane depth. Use `--units code` to display code units,
`--zoom` for a close-up of the plates, and `--frame` / `--save` for a static image.
`--full-domain` remains available and explicitly selects the default full view.
Runs without feeding display zero source and zero injected mass.

Export every saved frame as an H.264 MP4 with FFmpeg:

```sh
python3 plot_thruster_results.py output/thruster/injection/open/eta_0.001/thruster_injection_openbc_256x128_L36x24_rho_bg0.01_mdot0.01_J0.75128226_omega0.05_cycles10_rate0.02_etaBuffer0.001_outflow.h5 \
    --movie output/thruster/injection/open/eta_0.001/thruster_injection_openbc_256x128_L36x24_rho_bg0.01_mdot0.01_J0.75128226_omega0.05_cycles10_rate0.02_etaBuffer0.001_outflow.mp4 --fps 20
```

The completed [fiducial movie](output/thruster/injection/open/eta_0.001/thruster_injection_openbc_256x128_L36x24_rho_bg0.01_mdot0.01_J0.75128226_omega0.05_cycles10_rate0.02_etaBuffer0.001_outflow.mp4)
contains all 321 saved frames at 20 fps: 16.05 s at 1700 x 1200 pixels.
Its complete video stream was decoded successfully for verification.
`--frame` selects the initial preview frame; movie export always includes
the complete stored history. `--save` can also write a static image.

The solver's body force integrates gas pressure plus Maxwell stress over all
fluid/solid faces. The impermeable wall carries no advected mass flux. With
outward body normal `n`, the code-unit traction is

```text
F_body = -integral [ (p + B^2/2) I - B B ] dot n dA
T = -F_body,x
mean T = -Delta(solid_impulse_x) / Delta(time)
```

This includes magnetic pressure and tension; it is not a separate measurement
of reconnection thrust. The opposite force is the holding support's reaction.
The cumulative impulse includes every RK timestep, which is more accurate
for averages than integrating the sparse last-step force samples.

For a specified window, feed-based specific impulse is

```text
Isp = thrust impulse / (g0 * injected propellant mass)
    = mean thrust / (g0 * mean propellant mass flow)
g0 = 9.80665 m/s^2
```

Use SI quantities and integrate numerator and denominator over the same
window, rather than averaging instantaneous ratios. See
[NASA Glenn's specific impulse definition](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/specific-impulse/).
This is an apparent feed-based value: the finite ambient also participates,
even when the initial dense propellant load is disabled. A propellant
performance measurement needs a
balanced late-time inventory and a tracer or equivalent mass accounting to
separate injected material from ambient and any initial load. A finite-load
run without feeding uses the initial propellant mass expended, not a zero
mass-flow denominator. The viewer reports body traction; it does not add
unmodeled feed or return-circuit hardware forces.

## Numerical limitations

- Outflow boundaries use zero-gradient extrapolation and can still reflect waves.
- Resistivity is constant, scalar, and explicit.
- The default GLM resistive induction update omits the local resistive Poynting-flux
  correction, so it is not intended for detailed local Ohmic-heating studies.
- GLM cleaning controls but does not eliminate discrete magnetic divergence.
  In the 256 x 256 Orszag-Tang test at `t=0.5`,
  `dx*max(abs(divB)) = 3.39e-2` at localized shock cells and
  `dx*mean(abs(divB)) = 3.51e-4`.
- The antenna time table is stored in memory; fine sampling of large grids can
  be memory intensive.
- Stationary bodies use a cell-face staircase and a first-order electromagnetic
  interface. The prescribed-current branch also projects body current to nodal
  dual volumes. Material conductivity, viscous no-slip and body thermal evolution
  are absent.
- This is a compact research and teaching code, not a production MHD package.

## Repository layout

```text
src/main.cpp                 C++ finite-volume solver
run.py                       Python input preparation and launcher
input/                       Generic example problem definitions
plot_results.py              Harris/reconnection viewer
plot_antenna_results.py      Antenna-focused viewer
plot_thruster_results.py     Thruster source, density and force viewer
compare_orszag_tang.py       Athena++ benchmark comparison
reference/                   Compact benchmark reference data
Makefile                     Build and smoke tests
```
