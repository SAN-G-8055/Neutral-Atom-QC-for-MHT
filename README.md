# Interpretable cell detection and neutral-atom data association

This repository compares an exact classical solver with a simulated
neutral-atom solver inside one explicit tracking pipeline. Both solvers receive
the same immutable weighted conflict graph and return the same result schema.
All filtering, gating, Bayesian likelihood calculation, graph encoding, and
clustering happens once, outside the solver containers.

The former report/poster experiment and its monolithic global-hypothesis
scripts have been removed. The reusable detector, tracking mathematics, graph
encoder, classical optimizer, and QuTiP simulator now live in importable modules
with focused tests and a thin Jupyter interface.

## Layout

```text
src/neutral_atom_mht/
  detection.py              deterministic cell instance detection
  evaluation.py             gold-standard detection evaluation
  tracking/
    models.py               observations, retained tracks, local candidates
    filtering.py            Kalman prediction/update and track filtering
    gating.py               Mahalanobis and optional innovation-distance gates
    likelihood.py           shared Bayesian log-odds calculations
    interface.py            explicit prepare/compare/advance workflow
  graph.py                  immutable encoding, clustering, fingerprint, plot
  backends/
    base.py                 common SolverInput and SolverResult contract
    classical.py            exact classical MWIS backend
    neutral_atom.py         direct QuTiP Rydberg simulation
notebooks/                  package-only Jupyter interface
docs/tracking.md            equations, contracts, units, and limitations
tests/                      unit and integration tests
artifacts/detection/        reproducible sequence-01 detection benchmark
data/                       local source data, ignored by Git
```

## Install

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[test,quantum,notebook]"
pytest
```

The classical tracking and detection paths do not require QuTiP.
`requirements-detection-lock.txt` records the environment used for the curated
detection artifacts; `requirements-quantum-lock.txt` pins QuTiP 5.3.1 on top of
that numeric stack. Notebook and test-runner tooling follows the compatible
ranges in `pyproject.toml`; this project does not claim a fully locked Jupyter
or CI environment.

## One tracking interaction

At every frame, `TrackingInterface` calls these stages separately:

```text
predict -> gate -> calculate likelihoods -> filter candidates
        -> encode graph -> cluster -> solve
        -> Bayesian update -> filter tracks
```

An association hypothesis is only a local proposition that track `i` generated
observation `j` in the current frame. Graph edges mean two propositions reuse a
track or observation. Once one backend result is explicitly selected, the
candidates are discarded and one state per retained track remains. There are no
global-hypothesis families or backend-specific posterior marginals.

`prepare()` freezes the preprocessing output. `compare_prepared()` then gives
both backends byte-identical `SolverInput` fingerprints without mutating tracker
state. `advance()` applies the same Bayesian hit/miss equations after the caller
chooses a successful run. See [the tracking contract](docs/tracking.md) for the
formulas and validation rules.

## Classical versus simulated neutral atoms

Both backends return:

```text
SolverResult(
  schema_version, problem_id, input_fingerprint, backend,
  selected_ids, objective, feasible, status, runtime_seconds, diagnostics
)
```

The classical backend computes an exact maximum-weight independent set without
rounding weights. The neutral-atom backend builds the weighted Rydberg
Hamiltonian directly in QuTiP, using the serialized Pasqal Pulser
`WeightedAnalogDevice` profile and DMM-style local detuning. It returns one
highest-probability feasible bitstring; it does not retain a sampled family of
global hypotheses. This is a coherent, noiseless state-vector emulation, not a
live-QPU or hardware-noise model.

Quantum size or geometry limits are explicit statuses. The backend never
silently partitions the graph, repairs it into a different problem, or falls
back to the classical solver. The abstract graph and its physical embedding are
both auditable and visualizable.
Before evolution, a bounded exhaustive audit compares the abstract MWIS with
the ground state of the actual final diagonal Hamiltonian, including non-edge
`C6/r^6` tails. A topology-correct embedding is rejected if those physical
terms change the weighted optimum. Because the reference device imposes
register layout and filling constraints, results also state that
direct-coordinate QuTiP emulation does not establish hardware execution-readiness.

Open the interface with:

```powershell
jupyter lab notebooks/classical_vs_quantum.ipynb
```

The notebook uses a tiny deterministic example by default, shows every
preprocessing output, draws the conflict graph, and presents classical and
QuTiP results with identical columns.

## Detection input

Cell detection retains only **PhC-C2DL-PSC sequence 01** from the
[Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/):

- `01/`: 300 raw phase-contrast frames;
- `01_GT/TRA/`: human tracking markers used as detection gold.

Challenge Test Data, sequence 02, silver `ST`, and relabelled error masks are
not used. Prepare the retained files and reproduce the benchmark with:

```powershell
cell-detect prepare-data
cell-detect run
```

A detection event remains strictly
`(sequence, frame, detection_id, x_px, y_px, area_px, source)`, with zero-based
`x = column`, `y = row`. Tracking converts those events to `Observation`
objects through `observations_from_detections()`; the graph layer never reads a
TIFF or detector-specific object directly.

The full 300-frame detection benchmark has micro F1 `0.778` at the fixed 10 px
gate (precision `0.862`, recall `0.709`). Its exact events, matches, provenance,
and diagnostic figures remain under `artifacts/detection/sequence_01/`.
