# Interpretable cell detection and data association

This project turns microscopy images into cell observations, associates those
observations over time, and makes every intermediate decision available for
inspection. The object-oriented interface has two clear responsibilities:
the Hypothesis Processing Controller (`HPC`) owns image preprocessing and
tracking state, while a `Solver` chooses a consistent set of weighted
associations. `ClassicalSolver` is implemented and usable. `QuantumSolver` is
a Pulser/QuTiP implementation of the neutral-atom quantum adiabatic attempt.
Both solvers consume the same complete frame problem and hide their component
decomposition from the tracking controller.

No population of global hypotheses is retained. Candidate associations exist
for one frame, the selected result updates one state per track, and the local
candidates are then discarded.

## Model

```mermaid
flowchart LR
    Image["Image frame"] --> Observe["HPC.observe<br/>detect observations"]
    Observe --> Prepare["HPC<br/>predict, gate, weight"]
    Prepare --> Graph["HPC<br/>encode full frame graph"]
    Graph --> Input["one SolverInput"]

    Solver["Solver<br/>shared solve lifecycle"] -. inherited by .-> Components["ComponentSolver<br/>shared graph factoring"]
    Components -. inherited by .-> Classical["ClassicalSolver<br/>implemented"]
    Components -. inherited by .-> Quantum["QuantumSolver<br/>Pulser/QuTiP simulator"]
    Input --> Classical
    Input --> Quantum
    Classical --> Result["SolverResult"]
    Quantum --> Result

    Result --> Update["HPC.advance<br/>Bayesian and Kalman update"]
    Update --> State[("retained track state")]
    State -->|next frame| Prepare
```

The loop is deliberately visible: an image goes to `HPC`, immutable graph
problems go to a solver, and the common solver result returns to `HPC` before
the next frame is processed.

## Installation

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[test,notebook]"
python -m pytest
```

Install the optional Pulser simulation stack before using `QuantumSolver`:

```powershell
python -m pip install -e ".[quantum]"
```

The quantum dependency is loaded lazily. Importing the package and running the
classical path therefore do not require Pulser.

The real Pulser/QuTiP smoke test is opt-in because it runs the full embedding
search and 40,000 ns pulse:

```powershell
$env:NEUTRAL_ATOM_INTEGRATION="1"
python -m pytest tests/test_neutral_atom_integration.py -q
```

To reproduce the pinned numerical environment used for the curated detection
benchmark, constrain the runtime dependencies during installation:

```powershell
python -m pip install -c requirements-detection-lock.txt -e .
```

Launch the root-level interface with:

```powershell
jupyter lab user_notebook.ipynb
```

The notebook has five cells—imports, optional synthetic generation,
configuration, a single-frame run, and an optional multi-frame run. It uses the
real sequence-01 frame `data/PhC-C2DL-PSC/01/t000.tif` by default. Its import
cell adds the local `src/` directory explicitly, so it also works from an
uninstalled source checkout when opened from the repository root. The project
assumes real sequence TIFFs are already present under
`data/PhC-C2DL-PSC/`; there is no downloader or command-line interface.

Set `RUN_MANY_FRAMES = True` in the final cell to process the configured number
of consecutive images and plot active tracks and assigned observations. The
configuration cell also includes a commented `solver = QuantumSolver()` switch.
Install the quantum extra and begin with one frame before attempting a longer
simulated sequence.

## Object-oriented interface

The usual entry point is short:

```python
from neutral_atom_mht import ClassicalSolver, HPC, HPCConfig

hpc = HPC(HPCConfig())
solver = ClassicalSolver()
history = hpc.run_sequence(images, solver=solver)
```

The class uses Python's conventional `HPC` spelling; the package also exports
`hpc` as an exact lowercase class alias for interactive use.

`images` is any ordered iterable of two-dimensional image arrays. For each
frame, `run_sequence()` calls the same public stages that are available for
interactive use:

```text
observe image
  -> predict tracks
  -> gate observations
  -> calculate Bayesian association weights
  -> filter candidates
  -> encode the complete conflict graph
  -> create one immutable SolverInput
  -> Solver.solve(...)
  -> apply Bayesian/Kalman updates
  -> filter retained tracks
```

The main object responsibilities are:

| Object | Responsibility |
| --- | --- |
| `HPC` | Converts images to observations, exposes every preprocessing stage, owns retained track state, creates one full-frame solver input, validates its result, and advances the sequence. |
| `Solver` | Owns the shared `solve(SolverInput) -> SolverResult` lifecycle, including timing, objective calculation, and validation. It does not detect cells or mutate tracks. |
| `ComponentSolver` | Extends `Solver` with deterministic component factoring, a per-component capacity, and common graph diagnostics. |
| `ClassicalSolver` | Finds connected components internally and computes their exact maximum-weight independent sets, subject to its per-component size limit. |
| `QuantumSolver` | Finds connected components internally, embeds each supported component in a neutral-atom register, runs the Pulser/QuTiP simulation, decodes feasible samples to original graph-node IDs, and combines them into one result. |

For inspection, call `observe()`, `prepare_frame()`, `solve()`, and `advance()`
separately. `prepare_frame()` is read-only and exposes its predictions, gates,
weighted candidates, full graph, and solver input. `step()` is the
one-frame convenience method; `run_sequence()` repeats it over many images.
The notebook keeps the interface deliberately smaller: it prepares, solves,
and advances one real dataset frame without redefining package algorithms.

## Association model

An observation has a frame-local identifier, position `(x, y)`, and a `2 x 2`
measurement covariance. A retained track has one Kalman state
`(x, y, vx, vy)`, its covariance, existence log odds, and observation history.
An association candidate means only:

> retained track `i` generated observation `j` in the current frame.

Two candidates conflict when they reuse the same track or the same
observation. Each candidate is a graph node and each conflict is an edge, so a
valid association decision is an independent set.

### Gating

For innovation `nu` and innovation covariance `S`, the statistical gate uses

```text
d_squared = nu.T @ inv(S) @ nu.
```

The boundary is inclusive: a pair is retained when `d_squared` is no greater
than the configured threshold. An optional Euclidean innovation-distance gate
is a distinct second test; it is not described as a speed gate.

### Bayesian weights and updates

Track existence is stored as log odds:

```text
L = log(P(exists) / (1 - P(exists))).
```

For detection probability `P_D` and uniform clutter spatial density
`beta_FA`, a gated two-dimensional hit contributes

```text
d_hit = log(P_D) - log(beta_FA) - log(2*pi)
        - 0.5*log(det(S)) - 0.5*nu.T @ inv(S) @ nu.
```

A miss contributes:

```text
d_miss = log(1 - P_D).
```

Every posterior uses `L_new = L_old + d` and
`P_new = 1 / (1 + exp(-L_new))`. Because every unselected track receives the
same miss update, the graph weight is the incremental assignment benefit
`d_hit - d_miss`. These calculations belong to `HPC`, so every solver receives
identical weights and cannot apply a private probability update.

## Solver contract

The complete graph for one frame is frozen as a single `SolverInput` with a
canonical SHA-256 fingerprint. It may contain any number of disconnected
components. Each solver owns any component decomposition it needs and returns
one `SolverResult` with the common fields:

```text
schema_version, problem_id, input_fingerprint, solver_name,
selected_ids, objective, feasible, status, runtime_seconds, diagnostics
```

`HPC` validates that selected identifiers exist, do not conflict, and reproduce
the objective from the original graph weights. Only a validated successful
result may advance tracking state.

`ClassicalSolver` solves disconnected components independently behind this
boundary, returns their combined selection as `optimal`, and reports its
per-component size limit explicitly rather than changing the problem.

`QuantumSolver` applies the same internal decomposition, then runs the integrated
Pulser quantum-adiabatic sequence with the local QuTiP backend for each supported
component. Qubit positions are local implementation details: sampled bitstrings
are mapped back to the original graph-node IDs before the component selections
are combined. A usable simulated selection reports `completed`, not `optimal`,
because sampling does not prove the maximum-weight solution. Diagnostics retain
the component mappings, embedding information, and sample counts.

Singleton and complete-clique components are resolved directly by selecting the
highest positive-weight node, with a stable node-ID tie break. The original
Rabi-frequency heuristic requires at least one nonedge, so applying it to a
clique would not preserve its intended physical regime. This narrow topology
guard is not a general classical fallback; every other supported component uses
the Pulser simulation.

`NeutralAtomConfig` exposes `random_seed=0`, `mapping_tolerance=1e-6`,
`mapping_max_iterations=200_000`, `pulse_duration_ns=40_000`,
`interaction_scale=10.0`, and `qutip_cache_dir=data/.cache/qutip`.
`QuantumSolver(maximum_component_nodes=16)` owns the inherited component cap.
The cap protects the exponentially scaling state-vector simulation; an
oversized simulated component produces `unsupported_size` instead of a partial
tracking decision.

Likewise, unsupported negative simulation weights, a failed embedding, or no
valid feasible sample report `unsupported_weights`, `embedding_failed`, or
`no_feasible_sample` atomically.

`random_seed` scopes NumPy randomness across embedding and backend sampling as
far as the selected backend honors it; sampled, vendor-dependent results do not
carry an absolute reproducibility guarantee.

Pulser is imported only when this path is used. `NeutralAtomVisualizer` lives
in the same `neutral_atom.py` module as the solver, while its rendering remains
explicit and opt-in; solving itself does not display figures. The defaults use
`MockDevice` and `QutipBackendV2`, not Pasqal hardware.

Normal tracking calls `solve()`. To inspect the diagnostics from one simulation,
call `execute()` instead (not both for the same frame), then pass its run records
to a visualizer object:

```python
from neutral_atom_mht import NeutralAtomVisualizer, QuantumSolver

solver = QuantumSolver()
execution = solver.execute(solver_input)
if not execution.successful:
    raise RuntimeError(execution.diagnostics.get("message", execution.status))
run = execution.runs[0]
visualizer = NeutralAtomVisualizer()
visualizer.save_distribution(run, "quantum-samples.png")
if run.program is not None:
    visualizer.save_register(run.program, "quantum-register.png")
```

QuTiP's generated coefficient modules are redirected away from the repository
root into `data/.cache/qutip/`. The cache is local, ignored by Git, and can be
regenerated by the simulator.

## Synthetic data generation

Synthetic Cell Tracking Challenge-style sequences are available through the
same object-oriented package interface. `SyntheticDataConfig` describes a
reproducible scene, `SyntheticDataGenerator` renders it, and the returned
`SyntheticDataset` owns the generated paths and frame loaders:

```python
from neutral_atom_mht import SyntheticDataConfig, SyntheticDataGenerator

config = SyntheticDataConfig(
    noise=0.4,
    frame_count=20,
    object_count=30,
    seed=7,
)
dataset = SyntheticDataGenerator(config).generate()
image = dataset.load_frame(0)
labels = dataset.load_tracking_labels(0)
```

By default this creates `data/synthetic/SYN-MHT/`, with raw images in `01/`
and tracking labels in `01_GT/TRA/`. Generated datasets are local and ignored
by Git. To avoid mixed or stale sequences, generation refuses to overwrite a
nonempty named dataset directory. The generator deliberately owns only motion,
rendering, and CTC-style serialization; detection, association, benchmarking,
and visualization remain separate responsibilities. There is no synthetic-data
command-line interface.

The notebook exposes the same workflow without generating files during an
ordinary Run All. Set `GENERATE_SYNTHETIC_DATA = True` in its optional cell and
run that cell once, then set `USE_SYNTHETIC_DATA = True` in the configuration
cell. Switch generation back off after the dataset has been created because
existing nonempty sequences are intentionally not overwritten.

## Cell detection and gold-standard evaluation

The repository uses only **PhC-C2DL-PSC sequence 01** from the Cell Tracking
Challenge. The retained source data are:

- `01/`: 300 raw phase-contrast frames;
- `01_GT/TRA/`: human tracking markers used only as post-hoc detection gold.

Challenge test data, sequence 02, silver-standard `ST` annotations, and
relabelled error masks are not used. Source images are local and ignored by
Git. Place the retained files in this layout before opening the notebook or
calling `run_detection_benchmark()` from Python:

```text
data/PhC-C2DL-PSC/
  01/t000.tif ... t299.tif
  01_GT/TRA/man_track000.tif ... man_track299.tif
```

The detector is deterministic:

```text
Gaussian denoise
  -> broad-background subtraction
  -> Otsu-derived high-confidence seeds
  -> morphology and seed-area filtering
  -> connected low-threshold support
  -> nearest-seed assignment within each component
  -> final-area filtering
```

Gold labels never enter detection. A detection event is strictly one positive
final instance label in one frame, represented as:

```text
(sequence, frame, detection_id, x_px, y_px, area_px, source)
```

Coordinates are zero-based pixels with the origin at the upper left;
`x_px = column` and `y_px = row`. Prediction identifiers are frame-local. Gold
identifiers retain their source labels, but evaluation does not treat either
identifier as a cross-frame identity.

Predictions and gold centroids are matched independently per frame by
maximum-cardinality one-to-one assignment inside an inclusive Euclidean gate;
minimum total distance breaks cardinality ties. The primary figure of merit is
micro-averaged centroid F1 at a fixed 10 px gate:

```text
F1 = 2 * sum(TP) / (sum(predicted) + sum(gold)).
```

Across all 300 frames, the versioned run contains 50,622 true positives,
8,104 false positives, and 20,781 false negatives: precision `0.862`, recall
`0.709`, micro F1 `0.778`, and localization RMSE `3.692 px`. Exact events,
matches, frame metrics, 5/10/15 px sensitivity, figures, parameters, software
versions, and hashes are stored under `data/benchmark/`:

```text
data/benchmark/
  detections.csv
  gold_events.csv
  matches.csv
  per_frame_metrics.csv
  summary.json
  detections_overview.png
  performance_over_time.png
```

`summary.json` fingerprints all 300 source pairs. The raw-frame SHA-256 is
`46c15979d995a6e8f3bbbed78652965c7575fba8f4d49da87493903e051b90fa` and the
human-gold SHA-256 is
`4795100971222e24686c8dae8532c24d4c99d7401c08b418937b2968dd56f01b`.

## Project layout

```text
README.md                         this complete project guide
user_notebook.ipynb               five-cell real/synthetic sequence interface
pyproject.toml                    package and test configuration
requirements-detection-lock.txt  benchmark reproduction environment
data/
  benchmark/                      versioned detection evidence
  PhC-C2DL-PSC/                   local official sequence-01 images
  synthetic/                      locally generated synthetic datasets
  .cache/qutip/                   generated QuTiP coefficient cache
src/
  _version.py                      shared project version
  neutral_atom_mht.py             public package facade
  hpc.py                          stateful image-to-association controller
  solver.py                       solver lifecycle, component parent, and data contract
  classical_solver.py             implemented exact graph solver
  neutral_atom.py                 Pulser/QuTiP solver and opt-in diagnostics figures
  models.py                       observations, tracks, and candidates
  filtering.py                    Kalman prediction and update
  gating.py                       statistical and distance gates
  likelihood.py                   Bayesian weights and posterior updates
  graph.py                        graph encoding, clustering, and plotting
  detection.py                    deterministic cell detector
  evaluation.py                   gold-standard matching and metrics
  cell_data.py                    local sequence-01 paths and validation
  artifact_io.py                  benchmark serialization
  synthetic_data.py               OO synthetic sequence generator and loaders
  visualization.py               detection figures
  benchmark.py                    reproducible detection benchmark
tests/                            unit and end-to-end tests
NBQSS_Project_Real_Quantum_Attempt.ipynb
                                  original Pulser research notebook retained as provenance
```

Each source file begins with a plain-language module description stating what
it receives, what it produces, and which responsibility it deliberately does
not own. The flat modules preserve those boundaries without forcing users to
navigate nested solver or tracking packages.

## Current limitations

- Neutral-atom solving currently uses a local QuTiP state-vector simulator. Its
  sampled result is heuristic, depends on the graph embedding, and is subject to
  the configured component-size cap; it is not a hardware or optimality claim.
- The original quantum notebook is retained only as provenance. Synthetic data
  model controlled scenes and should not be treated as biological ground truth.
- The exact classical solver has exponential worst-case complexity and a
  declared node cap.
- Detection parameters are fixed for interpretability rather than learned or
  adapted per frame.
- Source TIFF files are not versioned; `summary.json` hashes establish the
  provenance of the published benchmark outputs.
