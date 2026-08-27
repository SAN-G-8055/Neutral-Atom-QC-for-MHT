# Neutral-atom quantum computing for multi-hypothesis tracking

This project turns microscopy images into cell observations, associates those
observations over time, and makes every intermediate decision available for
inspection. The Hypothesis Processing Controller (`HPC`) owns image
preprocessing and tracking state, while a `Solver` chooses a consistent set of
weighted associations. `ClassicalSolver` computes exact maximum-weight
independent sets; `QuantumSolver` is a Pulser/QuTiP neutral-atom quantum
adiabatic implementation of the same contract.

## Quick start

```powershell
python -m pip install -e ".[test,notebook,quantum]"
jupyter lab user_notebook.ipynb
```

The notebook is now a focused, short campaign. It varies one combined-noise
severity while retaining object count and random seed as design and replication
variables. Its 120 scenarios reuse 4,800 completed exact records from
`outputs/overnight/schema-2.1/overnight/benchmark.sqlite3`; old quantum records
are never imported. Stored exact selections restore the tracking trajectory
without running the classical optimizer again.

The new neutral-atom work uses a quota of three frames per severity/component-
size stratum and a 25-minute **forward-work checkpoint budget**. This is not a
hard wall-clock timeout: inexpensive replay and final export are outside the
budget, and an individual quantum call finishes atomically. Results are saved
under `outputs/overnight/schema-2.2/combined_quick/`; rerun the benchmark cell
to resume. The remaining cells generate eight figures, including the validated
frame-2 solver layout and combined-noise coverage, quality, and runtime plots.

To use the quantum solver, install the optional Pulser simulation stack:

```powershell
python -m pip install -e ".[quantum]"
```

The quantum dependency is loaded lazily; the classical path never requires
Pulser. Run the tests with `python -m pytest`. The real Pulser/QuTiP smoke
test is opt-in because it runs the full embedding search and 40,000 ns pulse:

```powershell
$env:NEUTRAL_ATOM_INTEGRATION="1"
python -m pytest tests/test_neutral_atom_integration.py -q
```

## Model

```mermaid
flowchart LR
    Image["Image frame"] --> Observe["HPC.observe<br/>detect observations"]
    Observe --> Prepare["HPC<br/>predict, gate, weight"]
    Prepare --> Graph["HPC<br/>encode full frame graph"]
    Graph --> Input["one SolverInput"]

    Solver["Solver<br/>shared solve lifecycle"] -. inherited by .-> Components["ComponentSolver<br/>shared graph factoring"]
    Components -. inherited by .-> Classical["ClassicalSolver<br/>exact"]
    Components -. inherited by .-> Quantum["QuantumSolver<br/>Pulser/QuTiP simulator"]
    Input --> Classical
    Input --> Quantum
    Classical --> Result["SolverResult"]
    Quantum --> Result

    Result --> Update["HPC.advance<br/>Bayesian and Kalman update"]
    Update --> State[("retained track state")]
    State -->|next frame| Prepare
```

An image goes to `HPC`, an immutable graph problem goes to a solver, and the
common solver result returns to `HPC` before the next frame is processed. No
population of global hypotheses is retained: candidate associations exist for
one frame, the selected result updates one state per track, and the local
candidates are then discarded.

## Object-oriented interface

```python
from neutral_atom_mht import ClassicalSolver, HPC, HPCConfig

hpc = HPC(HPCConfig())
solver = ClassicalSolver()
history = hpc.run_sequence(images, solver=solver)
```

`images` is any ordered iterable of two-dimensional image arrays. For
step-by-step inspection, call `observe()`, `prepare_frame()`, `solve()`, and
`advance()` separately; `prepare_frame()` is read-only and exposes its
predictions, gates, weighted candidates, full graph, and solver input.
`step()` is the one-frame convenience method.

| Object | Responsibility |
| --- | --- |
| `HPC` | Converts images to observations, owns retained track state, creates one full-frame solver input, validates its result, and advances the sequence. |
| `Solver` | Owns the shared `solve(SolverInput) -> SolverResult` lifecycle: timing, objective calculation, and validation. |
| `ComponentSolver` | Extends `Solver` with deterministic component factoring and a per-component capacity. |
| `ClassicalSolver` | Computes exact maximum-weight independent sets per connected component. |
| `QuantumSolver` | Embeds each supported component in a neutral-atom register, runs the Pulser/QuTiP simulation, and decodes feasible samples back to graph-node IDs. |

## Association model

An observation has a frame-local identifier, position `(x, y)`, and a `2 x 2`
measurement covariance. A retained track has one Kalman state
`(x, y, vx, vy)`, its covariance, existence log odds, and observation history.
A candidate means "track `i` generated observation `j` in this frame". Two
candidates conflict when they reuse the same track or observation, so a valid
association decision is an independent set in the conflict graph.

Gating keeps a pair when the Mahalanobis distance
`d_squared = nu.T @ inv(S) @ nu` is within the configured threshold (an
optional Euclidean innovation-distance gate is a distinct second test).
Track existence is stored as log odds; a gated hit contributes
`d_hit = log(P_D) - log(beta_FA) - log(2*pi) - 0.5*log(det(S)) - 0.5*d_squared`
and a miss contributes `d_miss = log(1 - P_D)`. Because every unselected track
receives the same miss update, the graph weight is the incremental benefit
`d_hit - d_miss`. These calculations belong to `HPC`, so every solver receives
identical weights.

## Solver contract

The complete frame graph is frozen as a single `SolverInput` with a canonical
SHA-256 fingerprint. Each solver owns any component decomposition it needs and
returns one `SolverResult`:

```text
schema_version, problem_id, input_fingerprint, solver_name,
selected_ids, objective, feasible, status, runtime_seconds, diagnostics
```

`HPC` validates that selected identifiers exist, do not conflict, and
reproduce the objective from the original graph weights; only a validated
result may advance tracking state.

`QuantumSolver` reports a usable simulated selection as `completed`, not
`optimal`, because sampling does not prove the maximum-weight solution.
Singleton and complete-clique components are resolved directly (the Rabi
heuristic requires at least one nonedge); every other supported component uses
the Pulser simulation. A converged interaction fit is accepted only when its
blockade graph exactly matches the logical component. If it does not, seeded
topology-aware refinements pull intended edges inside the blockade radius and
push nonedges outside it; an unrepairable component reports `embedding_failed`
before Pulser runs. `NeutralAtomConfig` exposes `random_seed`,
`mapping_tolerance`, `mapping_max_iterations`, `topology_restarts`,
`topology_safety_factor`, `pulse_duration_ns`, `interaction_scale`, and
`qutip_cache_dir`.
`QuantumSolver(maximum_component_nodes=16)` caps the exponentially scaling
state-vector simulation. Oversized components, negative weights, failed
embeddings, or missing feasible samples report `unsupported_size`,
`unsupported_weights`, `embedding_failed`, or `no_feasible_sample` atomically.
The defaults use `MockDevice` and `QutipBackendV2`, not Pasqal hardware.

To inspect one simulation, call `execute()` instead of `solve()` and pass its
run records to `NeutralAtomVisualizer`:

```python
from neutral_atom_mht import NeutralAtomVisualizer, QuantumSolver

solver = QuantumSolver()
execution = solver.execute(solver_input)
run = execution.runs[0]
visualizer = NeutralAtomVisualizer()
visualizer.save_distribution(run, "quantum-samples.png")
```

## Data

The optional real dataset is **PhC-C2DL-PSC sequence 01** from the Cell
Tracking Challenge. Source images are local and ignored by Git; place them as
`data/PhC-C2DL-PSC/01/t000.tif ... t299.tif` to have the notebook use them.

Synthetic Cell Tracking Challenge-style sequences are generated through the
same package interface. The notebook uses `QUANTUM_DEMO_DATA_CONFIG`, a
versioned preset with 8 frames, 4 objects, noise `0.1`, seed `0`, and
`256 x 320` images. It retains a real non-clique simulation while keeping
every observed conflict component at five nodes or fewer.

Custom sequences can use the same generator:

```python
from neutral_atom_mht import SyntheticDataConfig, SyntheticDataGenerator

config = SyntheticDataConfig(
    noise=0.4,
    frame_count=20,
    object_count=30,
    seed=7,
    dataset_name="SYN-MHT-CUSTOM-v1",
)
dataset = SyntheticDataGenerator(config).generate()
image = dataset.load_frame(0)
```

The `noise` setting remains a convenient coupled difficulty control. For
factorial experiments, its four effects can be replaced independently with
`speed_px_per_frame_override`, `detection_probability_override`,
`clutter_per_frame_override`, and `pixel_noise_sigma_override`. Large sweeps do
not need to materialize TIFFs:

```python
config = SyntheticDataConfig(
    noise=0.0,
    speed_px_per_frame_override=12.0,
    detection_probability_override=0.7,
    clutter_per_frame_override=18.0,
    pixel_noise_sigma_override=4.0,
)
for image, tracking_labels in SyntheticDataGenerator(config).iter_frames():
    ...
```

`build_synthetic_scenarios()` uses those overrides to isolate each degradation.
`run_overnight_benchmark()` streams the grid through the tracker, matches
detections to synthetic labels, screens exact MWIS results, selects a stable
axis/severity/size-stratified quantum sample, and commits each row to SQLite.
An optional `exact_checkpoint_source` imports compatible successful exact rows
from an older campaign, read-only, and replays their stored selections rather
than reoptimizing them.
Solver failures and frames outside the quantum component cap are retained as
results rather than terminating the campaign. Notebook heatmaps annotate
prepared-frame coverage and quantum sample counts. Treat partial cells as
descriptive only: the quantum quota is stratified by difficulty axis, severity,
and component size, but not by density, seed, or frame.

The joined records also evaluate retained tracks against synthetic identities:
tracking recall and precision, spatial error, identity correctness, ID switches,
and fragmentations remain available alongside detector and solver metrics.

The notebook streams its synthetic frames in memory. Its first figure also
shows `data/PhC-C2DL-PSC/01/t000.tif` when that optional real frame is installed;
otherwise the real-data panel is explicitly marked unavailable.

## Project layout

```text
README.md                this complete project guide
user_notebook.ipynb      the run-everything notebook interface
pyproject.toml           package and test configuration
data/                    local datasets (ignored by Git)
src/
  neutral_atom_mht.py    public package facade
  hpc.py                 stateful image-to-association controller
  solver.py              solver lifecycle, component parent, and data contract
  classical_solver.py    exact graph solver
  neutral_atom.py        Pulser/QuTiP solver and opt-in diagnostics figures
  overnight_benchmark.py streamed, resumable exact/quantum experiment campaign
  models.py              observations, tracks, and candidates
  filtering.py           Kalman prediction and update
  gating.py              statistical and distance gates
  likelihood.py          Bayesian weights and posterior updates
  graph.py               graph encoding, clustering, and plotting
  detection.py           deterministic cell detector
  cell_data.py           local sequence-01 paths and loading
  synthetic_data.py      synthetic sequence generator and loaders
tests/                   unit and end-to-end tests
```

Each source file begins with a plain-language module description stating what
it receives, what it produces, and which responsibility it deliberately does
not own.

## Current limitations

- Neutral-atom solving uses a local QuTiP state-vector simulator. Its sampled
  result is heuristic, depends on the graph embedding, and is subject to the
  configured component-size cap; it is not a hardware or optimality claim.
- The exact classical solver has exponential worst-case complexity and a
  declared node cap.
- Synthetic data model controlled scenes and should not be treated as
  biological ground truth.
- Detection parameters are fixed for interpretability rather than learned or
  adapted per frame.
