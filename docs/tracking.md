# Tracking and solver contract

This stage keeps exactly one state for each retained track. Candidate
associations exist for one frame only and are discarded after a solver selects
one consistent set. There is no population of global hypotheses, no family
tree, and no backend-dependent probability pruning.

## One explicit interaction

Every interaction with either backend follows the same public stages:

```text
observations
  -> predict_tracks()                    Kalman prediction
  -> gate_observations()                 Mahalanobis and optional innovation-distance gate
  -> calculate_association_hypotheses()  shared hit likelihood and log-odds
  -> filter_association_hypotheses()     declared minimum weight
  -> encode_conflict_graph()             one node per local association
  -> cluster_graph()                     connected components
  -> backend.solve(SolverInput)           classical or neutral_atom_qutip
  -> apply_bayesian_updates()             shared hit/miss update
  -> filter_tracks()                      declared posterior/miss/cap rules
```

`TrackingInterface.prepare()` performs everything through clustering without
changing state. The resulting `SolverInput` is immutable and has a canonical
SHA-256 fingerprint. `compare_prepared()` passes inputs with those same
fingerprints to both backends. Only `advance()` mutates the tracker, and the
caller must explicitly choose which successful backend run to use.

The statistical gate is inclusive at its declared Mahalanobis-squared
threshold. `maximum_innovation_distance`, when configured, is a separate hard
Euclidean bound on the observation residual. It is intentionally not called a
speed limit: elapsed time belongs to the preceding Kalman prediction step.

## Bayesian association weight

The retained track weight is a log odds,

```text
L = log(P(exists) / (1 - P(exists))).
```

For a gated two-dimensional observation with innovation `nu`, innovation
covariance `S`, detection probability `P_D`, and uniform clutter spatial
density `beta_FA`, the hit increment is

```text
d_hit = log(P_D) - log(beta_FA) - log(2*pi)
        - 0.5*log(det(S)) - 0.5*nu.T @ inv(S) @ nu.
```

A miss uses

```text
d_miss = log(1 - P_D).
```

The update is always `L_new = L_old + d`, and the reported existence
probability is `sigmoid(L_new)`. An unselected track receives `d_miss`, so that
term is the common assignment baseline. A graph vertex is weighted by the
positive improvement `d_hit - d_miss`; this is exactly the change in joint
log score produced by selecting that association. All of these calculations
happen outside both solver containers, so the classical and quantum objectives
are identical.

## Conflict graph

One graph node means “associate this existing track with this observation in
this frame.” Two nodes share an edge when they reuse either the same track or
the same observation. An independent set is therefore a valid one-to-one local
association decision.

[`neutral_atom_mht.graph`](../src/neutral_atom_mht/graph.py) owns the immutable
encoding, fingerprint, connected-component clustering, logical layout, and PNG
visualization. It deliberately contains no solver or quantum code.

## Common backend output

Both containers return `SolverResult` with the same fields:

```text
schema_version, problem_id, input_fingerprint, backend,
selected_ids, objective, feasible, status, runtime_seconds, diagnostics
```

The interface recomputes `objective` with `math.fsum` over the original
Bayesian graph weights and rejects conflicting or unknown selected IDs.
Quantum-only physical details remain under `diagnostics`; they never alter the
common objective.

The exact classical backend returns one optimum and fails transparently above
its declared exponential size limit. The QuTiP backend returns one
highest-probability feasible bitstring and discards the rest of the evolved
distribution with status `simulated`; it performs no shot sampling. It reports
`unsupported_size`, `embedding_error`, or
`dependency_missing` instead of silently switching to a classical algorithm.

## Neutral-atom simulation

The simulator uses QuTiP directly with the ground/Rydberg basis
`|g> = |0>`, `|r> = |1>` and occupation `n = |r><r|`:

```text
H/hbar = sum(i<j) C6 / r_ij^6 * n_i*n_j
         + Omega(t)/2 * sum(i) X_i
         - sum(i) [delta(t) + epsilon_i*delta_DMM(t)] * n_i.
```

Weights use the Detuning Map Modulator convention from Pasqal's official
[MWIS tutorial](https://docs.pasqal.com/pulser/tutorials/mwis/):
`epsilon_i = 1 - w_i/max(w)`. Lower-weight atoms receive more negative local
detuning, while the maximum-weight atom receives none.

The serialized reference profile follows Pasqal's published Pulser
[`WeightedAnalogDevice`](https://docs.pasqal.com/pulser/apidoc/_autosummary/pulser.WeightedAnalogDevice/):

| Parameter | Reference value |
| --- | ---: |
| Rydberg level | 75 |
| `C6/hbar` | 12,241,414.53 rad/us um^6 |
| Minimum atom spacing | 5 um |
| Maximum atoms | 256 |
| Maximum radial distance | 80 um |
| Maximum sequence duration | 6 us |
| Maximum runs | 500 |
| Maximum global Rabi amplitude | 4*pi rad/us |
| Maximum absolute global detuning | 20*pi rad/us |
| Minimum average global Rabi amplitude | 0.6*pi rad/us |
| DMM bottom detuning | -20*pi rad/us |
| Required layout traps | 150--512 |
| Required layout filling | 0.35--0.5 |

These are public reference-device constraints, not a claim about a particular
live Pasqal QPU. The implementation records them in every quantum result. The
device requires a register layout with declared trap and filling constraints;
this direct-coordinate QuTiP emulator does not validate layout selection or
hardware execution-readiness, and states that limitation explicitly in its
diagnostics.
QuTiP's official documentation describes the
[time-dependent solver interface](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-time.html)
used here.

The state-vector cost scales as `2**N`, so the simulator cap defaults well below
the hardware atom count. This is a coherent, noiseless state-vector simulation;
it is not a hardware noise or sampling model. The geometric encoder audits all
expected edges, non-edges, minimum spacing, and radial extent. It then
exhaustively compares the abstract MWIS with the actual final diagonal
Hamiltonian, including every `C6/r^6` interaction tail and every DMM reward, up
to the bounded simulation cap. Diagnostics report the largest non-edge
interaction, the abstract optimum, the physical ground state, and weighted
objective fidelity. A topology-correct layout whose residual interactions
change the optimum is therefore returned as `embedding_error`, not simulated as
if it represented the original graph.

## Notebook

[`notebooks/classical_vs_quantum.ipynb`](../notebooks/classical_vs_quantum.ipynb)
is a thin interface over these package APIs. It contains no algorithm
definitions or mutable module globals. Its small example exposes every
preprocessing object, visualizes the encoded graph, prints the common result
table, and advances only the explicitly selected backend.
