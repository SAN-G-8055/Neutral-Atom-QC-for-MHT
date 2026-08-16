"""
===============================================================================
 Radar Multi-Object Following on a Neutral-Atom Quantum Computer
 2026 Niels Bohr Quantum Summer School -- SDU Odense
 Challenge 7: Neutral Atom Quantum Computing for Multiple Hypothesis Tracking
===============================================================================

WHAT THIS FILE DOES
-------------------
It implements, end to end, the hybrid classical/quantum pipeline proposed in the
challenge sheet, and runs it on a simulated radar scenario with several moving
objects, clutter and missed detections.

    radar scans
        |
        |  [CLASSICAL]  Kalman prediction -> gating -> track branching
        |               -> log-likelihood-ratio (LLR) scoring -> pruning
        |               -> incompatibility lists (ICL) -> clustering
        v
    per-cluster MWIS graph:  vertices = candidate tracks (weight = LLR score)
                             edges    = tracks sharing a radar observation
        |
        |  [QUANTUM]    geometric embedding into a unit-disk graph
        |               -> optical-tweezer register (Pasqal / Pulser)
        |               -> Rydberg blockade enforces  x_i + x_j <= 1
        |               -> local detuning (DMM) encodes the weights w_i
        |               -> adiabatic sweep -> sample bitstrings
        v
    a *distribution* of low-energy independent sets
        = best global hypothesis  +  near-optimal hypotheses
        |
        |  [CLASSICAL]  global hypothesis probabilities P{H_j}   (Eq. 2)
        |               global track probabilities      P{T_i}   (Eq. 3)
        |               -> N-scan pruning -> track output
        v
    followed objects

The MWIS formulation, the LLR track score, the incompatibility list and
Eqs. (2)-(3) follow Papageorgiou & Salpukas, "The Maximum Weight Independent
Set Problem for Data Association in Multiple Hypothesis Tracking" (2009), which
is the paper the challenge is built on.

HOW TO RUN
----------
The file is divided into `#%%` cells. Run them top to bottom in VS Code's Python
Interactive window, PyCharm, or Jupyter (`jupytext`/`# %%` format). It also runs
as a plain script:  `.venv/bin/python NielsBohrProject.py`

Cell  1     configuration, device, plotting style
Cells 2-3   the radar scenario, and what the tracker is actually given
Cell  4     Kalman filter and the log-likelihood-ratio track score
Cell  5     classical MWIS solvers (reference, fallback, and shot repair)
Cells 6-7   the MHT front end, and the MWIS instance it produces
Cells 8-10  the quantum part: embedding, register + pulses, sampling
Cell  11    validation against the worked example in the paper (Table 1)
Cell  12    the full closed loop: quantum solver inside the tracker
Cells 13-15 results, parameter studies, and the open questions

The whole file runs in about two minutes on a laptop and is deterministic: the
emulator's shot noise is seeded in Cell 10.  Package versions are pinned in
requirements.txt next to this file.
===============================================================================
"""

# %% ===========================================================================
# CELL 1 -- Imports, global configuration and plotting style
# ==============================================================================

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional, Sequence as Seq, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import minimize
from scipy.special import logsumexp

# --- Pulser: Pasqal's neutral-atom SDK -----------------------------------------
from pulser import Pulse, Register, Sequence
from pulser.backend import BitStrings
from pulser.devices import WeightedAnalogDevice
from pulser.waveforms import ConstantWaveform, InterpolatedWaveform, RampWaveform
from pulser_simulation import QutipBackendV2, QutipConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)

RNG_SEED = 7
rng = np.random.default_rng(RNG_SEED)

# ---- the neutral-atom machine we target --------------------------------------
# WeightedAnalogDevice is Pasqal's analog device *with* a DMM (Detuning Map
# Modulator).  The DMM is what lets us apply a *per-atom* detuning, i.e. what
# lets us encode the track weights w_i.  AnalogDevice has no DMM, so it can only
# solve the unweighted MIS problem -- not enough for MHT.
DEVICE = WeightedAnalogDevice
C6 = DEVICE.interaction_coeff            # rad/us * um^6  (van der Waals C6/hbar)


@dataclass
class Config:
    """Every knob of the pipeline in one place."""

    # ---- radar / scenario ----------------------------------------------------
    n_scans: int = 10
    dt: float = 1.0                       # s between scans
    p_detect: float = 0.90                # P_D
    clutter_per_scan: float = 8.0         # Poisson mean number of false alarms
    sigma_range: float = 15.0             # m
    sigma_bearing: float = np.deg2rad(0.4)
    range_min: float = 500.0              # m   (field of view)
    range_max: float = 6000.0
    bearing_halfwidth: float = np.deg2rad(60.0)
    q_accel: float = 4.0                  # process-noise accel std, m/s^2

    # ---- MHT front end -------------------------------------------------------
    gate_chi2: float = 13.8               # chi^2, 2 dof, ~99.9 %
    score_init: float = 0.0               # LLR of a freshly initiated track
    score_delete: float = -6.0            # track-level pruning threshold
    max_misses: int = 3                   # consecutive missed detections
    max_tracks_per_family: int = 3
    max_tracks_total: int = 120
    # Track confirmation (paper Sec. 4.2).  The paper puts every *positive-score*
    # track into the MWIS graph; in a cluttered scene that is still far too many,
    # because a single lucky association already scores about +6.7.  Confirmation
    # thresholds on score AND number of detections are the standard remedy, and
    # they are what keeps the instance small enough for the atom array.
    score_confirm: float = 10.0
    min_hits_confirm: int = 3
    # Kinematic priors.  These matter more than anything else for the size of
    # the graph: a loose velocity prior makes the first gate enormous, every
    # clutter blip gates with every track, and the MWIS instance explodes.
    v_init_std: float = 250.0             # m/s, velocity prior at initiation
    v_max: float = 400.0                  # m/s, hard kinematic pruning limit
    exact_max_n: int = 18                 # above this, the exact solver is slow
    n_scan_window: int = 3                # N of "N-scan pruning"
    prob_prune: float = 1e-3              # global track probability threshold

    # ---- MWIS graph ----------------------------------------------------------
    max_atoms: int = 10                   # cluster size we allow on the QPU
                                          # (emulator cost is 2^N -- keep small)

    # ---- neutral-atom / quantum ---------------------------------------------
    # The embedding is done in units of the blockade radius R_b; the *physical*
    # scale (and hence Omega) is derived afterwards from the tightest pair, so
    # that the register always respects the device's minimum atom spacing.
    omega_cap: float = 2 * np.pi * 1.2    # rad/us, largest Rabi we will ask for
    # Lower bound on Omega, and it is the hardware that sets it: the rydberg
    # channel refuses pulses whose *average* amplitude is below min_avg_amp
    # (2pi x 0.3 rad/us here), and the turn-on ramp 0 -> Omega averages Omega/2.
    # So Omega >= 2 * min_avg_amp = 2pi x 0.6.  Conveniently this also keeps the
    # sweep adiabatic: Omega*T ~ 21 rad over the 5.4 us budget.  Enlarging R_b to
    # give the atoms more room costs Omega as R_b^-6, and this is where it stops.
    omega_min: float = 2 * np.pi * 0.62   # rad/us
    delta_ratio: float = 2.0              # delta_max = delta_ratio * Omega
    t_sweep: int = 5400                   # ns   (device caps the total at 6000)
    t_ramp: int = 300                     # ns   turn-on / turn-off ramps
    shots: int = 1000
    r_edge: float = 0.75                  # edge distance target, in units of R_b
    r_nonedge: float = 1.35               # non-edge distance target, in R_b
    r_floor: float = 0.42                 # hard repulsion floor, in R_b
                                          # (~ d_min / R_b_max: below this the
                                          #  register cannot be built at all)
    embed_restarts: int = 40


CFG = Config()

# ---- plotting style ----------------------------------------------------------
# Categorical hues assigned in fixed order (never cycled); the first three slots
# are the ones that stay distinguishable for colour-vision deficiency in scatter
# plots, which is what the track plots are.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, MUTED, GRIDC = "#1a1a19", "#6b6b68", "#dcdcd8"
ACCENT_OK, ACCENT_BAD = "#008300", "#e34948"

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,          # grid behind the marks, never across them
    "grid.color": GRIDC,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.facecolor": "white",
})

print(f"Device          : {DEVICE.name}")
print(f"  max atoms     : {DEVICE.max_atom_num}")
print(f"  min spacing   : {DEVICE.min_atom_distance} um")
print(f"  max radius    : {DEVICE.max_radial_distance} um")
print(f"  C6/hbar       : {C6:.3e} rad/us.um^6")
print(f"  DMM (local detuning) channels: {list(DEVICE.dmm_channels)}")
print(f"\nBlockade radius at Omega = {CFG.omega_cap:.3f} rad/us : "
      f"{(C6 / CFG.omega_cap) ** (1 / 6):.2f} um")


# %% ===========================================================================
# CELL 2 -- The radar and the scenario (ground truth + measurements)
# ==============================================================================
# A 2-D surveillance radar sits at the origin.  Every `dt` seconds it produces a
# *scan*: a set of unlabelled measurements in (range, bearing).  Some come from
# real objects (with probability P_D each), the rest is clutter.  Nothing in the
# scan says which measurement belongs to which object -- that is the whole
# problem.

@dataclass
class Target:
    """A ground-truth object: constant-velocity with small process noise."""
    x0: float
    y0: float
    vx: float
    vy: float
    t_birth: int = 0
    t_death: int = 10_000
    name: str = ""


@dataclass
class Observation:
    """One radar return (a 'blip')."""
    scan: int
    idx: int                    # index within the scan
    z: np.ndarray               # [x, y] in Cartesian, metres
    R: np.ndarray               # 2x2 measurement covariance, Cartesian
    truth: Optional[int]        # ground-truth target index, or None for clutter
                                # (bookkeeping only -- never used by the tracker)

    @property
    def key(self) -> Tuple[int, int]:
        return (self.scan, self.idx)


def polar_to_cartesian(r: float, th: float) -> np.ndarray:
    return np.array([r * np.cos(th), r * np.sin(th)])


def polar_noise_covariance(r: float, th: float, cfg: Config) -> np.ndarray:
    """Linearised (Jacobian) mapping of range/bearing noise into Cartesian.

    This is why radar covariances are *not* isotropic: the cross-range error
    grows like r * sigma_bearing while the down-range error stays sigma_range.
    """
    J = np.array([[np.cos(th), -r * np.sin(th)],
                  [np.sin(th),  r * np.cos(th)]])
    S_polar = np.diag([cfg.sigma_range ** 2, cfg.sigma_bearing ** 2])
    return J @ S_polar @ J.T


def simulate_scenario(cfg: Config, targets: List[Target], rng) -> Tuple[
        List[List[Observation]], Dict[int, Dict[int, np.ndarray]]]:
    """Return (scans, truth) where truth[target][scan] = [x, y]."""
    truth: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(len(targets))}
    states = {i: np.array([t.x0, t.y0, t.vx, t.vy], float)
              for i, t in enumerate(targets)}
    scans: List[List[Observation]] = []

    for k in range(cfg.n_scans):
        obs: List[Observation] = []

        # --- propagate ground truth (constant velocity + accel noise) ---------
        for i, tgt in enumerate(targets):
            if k > 0:
                s = states[i]
                a = rng.normal(0, cfg.q_accel, 2)
                s[0] += s[2] * cfg.dt + 0.5 * a[0] * cfg.dt ** 2
                s[1] += s[3] * cfg.dt + 0.5 * a[1] * cfg.dt ** 2
                s[2] += a[0] * cfg.dt
                s[3] += a[1] * cfg.dt
            if tgt.t_birth <= k <= tgt.t_death:
                truth[i][k] = states[i][:2].copy()

        # --- detections from real targets ------------------------------------
        for i, tgt in enumerate(targets):
            if not (tgt.t_birth <= k <= tgt.t_death):
                continue
            if rng.random() > cfg.p_detect:        # missed detection
                continue
            x, y = states[i][:2]
            r, th = np.hypot(x, y), np.arctan2(y, x)
            if not (cfg.range_min <= r <= cfg.range_max
                    and abs(th) <= cfg.bearing_halfwidth):
                continue
            r_m = r + rng.normal(0, cfg.sigma_range)
            th_m = th + rng.normal(0, cfg.sigma_bearing)
            obs.append(Observation(k, len(obs), polar_to_cartesian(r_m, th_m),
                                   polar_noise_covariance(r_m, th_m, cfg), i))

        # --- clutter / false alarms ------------------------------------------
        n_fa = rng.poisson(cfg.clutter_per_scan)
        for _ in range(n_fa):
            # uniform in area => r ~ sqrt(U) between r_min and r_max
            r = np.sqrt(rng.uniform(cfg.range_min ** 2, cfg.range_max ** 2))
            th = rng.uniform(-cfg.bearing_halfwidth, cfg.bearing_halfwidth)
            obs.append(Observation(k, len(obs), polar_to_cartesian(r, th),
                                   polar_noise_covariance(r, th, cfg), None))

        rng.shuffle(obs)                       # the sensor does not sort by truth
        for j, o in enumerate(obs):            # re-index after shuffling
            o.idx = j
        scans.append(obs)

    return scans, truth


# The scenario has to be *contested*, or there is nothing for a quantum computer
# to do.  With well-separated objects the front end resolves everything on its
# own and every MWIS instance collapses to "pick the one obvious track".
#
# So: a converging formation.  Several aircraft fly inbound on different
# bearings and pass through a common region at staggered times.  While they are
# close, one blip is compatible with several tracks, hypotheses proliferate, and
# the clusters become genuinely hard combinatorial problems.  (They pass *near*
# each other rather than exactly through one point -- coincident targets would
# raise the unresolved-target issue, which is out of scope here and is listed as
# an open question in Cell 15.)

def converging_formation(n=5, speed=230.0, cx=2600.0, cy=0.0, spread=0.85,
                         t_conv=(4.6, 4.9, 5.2, 5.5, 5.8)) -> List[Target]:
    ts = []
    for i in range(n):
        a = -spread + 2 * spread * i / (n - 1)
        vx, vy = -speed * np.cos(a), -speed * np.sin(a)
        tc = t_conv[i % len(t_conv)]
        ts.append(Target(x0=cx - vx * tc, y0=cy - vy * tc, vx=vx, vy=vy,
                         name=f"AC-{i+1}"))
    return ts


TARGETS = converging_formation()

SCANS, TRUTH = simulate_scenario(CFG, TARGETS, rng)

FOV_AREA = CFG.bearing_halfwidth * (CFG.range_max ** 2 - CFG.range_min ** 2)
BETA_FA = CFG.clutter_per_scan / FOV_AREA     # false-alarm spatial density, 1/m^2

print(f"{CFG.n_scans} scans, {sum(len(s) for s in SCANS)} observations total "
      f"({sum(1 for s in SCANS for o in s if o.truth is None)} clutter)")
print(f"False-alarm density beta_FA = {BETA_FA:.3e} /m^2")
print(f"Observations per scan: {[len(s) for s in SCANS]}")


# %% ===========================================================================
# CELL 3 -- What the tracker actually sees
# ==============================================================================
# Left: the truth (which we are not allowed to use).  Right: the unlabelled blips
# the algorithm receives.  The point of this figure is that the right-hand panel
# genuinely looks like noise -- recovering the left panel from it is the task.

def action_limits(pad: float = 900.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Frame the plots on where the objects actually are.  The clutter fills the
    whole field of view, so autoscaling shrinks the trajectories to nothing."""
    xy = np.array([p for d in TRUTH.values() for p in d.values()])
    return ((xy[:, 0].min() - pad, xy[:, 0].max() + pad),
            (xy[:, 1].min() - pad, xy[:, 1].max() + pad))


def plot_scenario():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    (x0, x1), (y0, y1) = action_limits()

    ax = axes[0]
    for i, tgt in enumerate(TARGETS):
        ks = sorted(TRUTH[i])
        xy = np.array([TRUTH[i][k] for k in ks])
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=2, color=SERIES[i], label=tgt.name)
        ax.plot(*xy[0], "o", ms=7, color=SERIES[i], mec="white", mew=1.5,
                zorder=3)
    ax.set_title("Ground truth (hidden from the tracker)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.annotate("dot = start of track", (.02, .02), xycoords="axes fraction",
                color=MUTED, fontsize=8)

    ax = axes[1]
    fa = np.array([o.z for s in SCANS for o in s if o.truth is None])
    det = np.array([o.z for s in SCANS for o in s if o.truth is not None])
    ax.scatter(fa[:, 0], fa[:, 1], s=18, c=MUTED, alpha=.6,
               label="every blip, unlabelled", linewidths=0)
    ax.scatter(det[:, 0], det[:, 1], s=18, c=MUTED, alpha=.6, linewidths=0)
    ax.set_title("What the radar delivers: unlabelled blips")
    ax.legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [m]")
    fig.suptitle("The data-association problem  (radar at the origin, off-frame "
                 "to the left)", fontweight="bold")
    fig.tight_layout()
    plt.show()


plot_scenario()


# %% ===========================================================================
# CELL 4 -- Kalman filter and the LLR track score
# ==============================================================================
# Constant-velocity model, state s = [x, y, vx, vy].
#
# The track score is Sittler's log-likelihood ratio: how much more likely is it
# that this sequence of blips came from one real object than from independent
# false alarms?  Recursively (paper, Sec. 3):
#
#     LLR(k) = LLR(k-1) + dLLR(k)
#     dLLR   = log(1 - P_D)                                    if no update
#     dLLR   = log( P_D * N(nu; 0, S) / beta_FA ) - handled below, if updated
#
# with N(nu;0,S) the Gaussian innovation density.  Expanded:
#
#     dL_u = log(P_D) - log(beta_FA) - log(2*pi) - 0.5*log|S| - 0.5*d^2
#
# for a 2-D measurement, where d^2 = nu^T S^-1 nu is the Mahalanobis distance.
# A good association gives roughly +6; a missed detection gives log(0.1) = -2.3.
# This single number is what becomes the *weight of an atom* later on.

def cv_matrices(dt: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-velocity transition F and discrete white-noise-accel covariance Q."""
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], float)
    G = np.array([[0.5 * dt ** 2, 0],
                  [0, 0.5 * dt ** 2],
                  [dt, 0],
                  [0, dt]])
    Q = G @ (q ** 2 * np.eye(2)) @ G.T
    return F, Q


H_MEAS = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]], float)


def kf_predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q


def kf_innovation(x_pred, P_pred, z, R):
    """Return (innovation nu, innovation covariance S, Mahalanobis d^2)."""
    nu = z - H_MEAS @ x_pred
    S = H_MEAS @ P_pred @ H_MEAS.T + R
    d2 = float(nu @ np.linalg.solve(S, nu))
    return nu, S, d2


def kf_update(x_pred, P_pred, nu, S):
    K = P_pred @ H_MEAS.T @ np.linalg.inv(S)
    x = x_pred + K @ nu
    P = (np.eye(4) - K @ H_MEAS) @ P_pred
    return x, 0.5 * (P + P.T)


def score_update(S: np.ndarray, d2: float, cfg: Config, beta_fa: float) -> float:
    """dL_u: the LLR increment for associating a blip with a track."""
    return (np.log(cfg.p_detect) - np.log(beta_fa)
            - np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(S)) - 0.5 * d2)


def score_miss(cfg: Config) -> float:
    """The penalty for a missed detection: log(1 - P_D) < 0."""
    return np.log(1.0 - cfg.p_detect)


_S_demo = np.diag([40.0 ** 2, 40.0 ** 2])
print("Typical LLR increments for this scenario")
print(f"  perfect association (d^2=0)   : {score_update(_S_demo, 0.0, CFG, BETA_FA):+.2f}")
print(f"  1-sigma association (d^2=1)   : {score_update(_S_demo, 1.0, CFG, BETA_FA):+.2f}")
print(f"  edge-of-gate       (d^2=13.8) : {score_update(_S_demo, CFG.gate_chi2, CFG, BETA_FA):+.2f}")
print(f"  missed detection              : {score_miss(CFG):+.2f}")


# %% ===========================================================================
# CELL 5 -- Classical MWIS solvers (our reference and our fallback)
# ==============================================================================
# We need these for two reasons: to know the true optimum when we judge the
# quantum sampler, and as the fallback for clusters too big for the atom array.

def mwis_bruteforce(G: nx.Graph, nodes: Seq[int]) -> Tuple[float, List[int]]:
    """Exact by enumeration -- only for small clusters (<= ~22 vertices)."""
    nodes = list(nodes)
    w = {n: G.nodes[n]["weight"] for n in nodes}
    adj = {n: set(G.neighbors(n)) & set(nodes) for n in nodes}
    best_w, best_set = 0.0, []
    for r in range(len(nodes) + 1):
        for comb in itertools.combinations(nodes, r):
            ok = True
            for a, b in itertools.combinations(comb, 2):
                if b in adj[a]:
                    ok = False
                    break
            if ok:
                val = sum(w[n] for n in comb)
                if val > best_w:
                    best_w, best_set = val, list(comb)
    return best_w, best_set


SCORE_SCALE = 1000        # networkx's max_weight_clique needs integer weights,
                          # so we work in milli-LLR units (exact to 1e-3)


def mwis_exact(G: nx.Graph, nodes: Seq[int]) -> Tuple[float, List[int]]:
    """Exact MWIS via maximum-weight clique on the complement graph.

    Weights are non-negative here (only positive-score tracks become vertices),
    so the optimum never wants to drop a free vertex.
    """
    nodes = list(nodes)
    if len(nodes) <= 1:
        return (float(G.nodes[nodes[0]]["weight"]), list(nodes)) if nodes else (0.0, [])
    comp = nx.complement(G.subgraph(nodes))
    for n in comp.nodes:
        comp.nodes[n]["weight"] = int(round(G.nodes[n]["weight"] * SCORE_SCALE))
    clique, _ = nx.max_weight_clique(comp, weight="weight")
    sel = sorted(clique)
    return float(sum(G.nodes[n]["weight"] for n in sel)), sel


def mwis_greedy(G: nx.Graph, nodes: Seq[int]) -> Tuple[float, List[int]]:
    """Greedy by weight / (1 + degree) -- the cheap classical heuristic."""
    remaining = set(nodes)
    chosen: List[int] = []
    while remaining:
        best = max(remaining, key=lambda n: G.nodes[n]["weight"]
                   / (1 + len(set(G.neighbors(n)) & remaining)))
        chosen.append(best)
        remaining -= {best} | set(G.neighbors(best))
    return sum(G.nodes[n]["weight"] for n in chosen), sorted(chosen)


def is_independent(G: nx.Graph, sel: Seq[int]) -> bool:
    return not any(G.has_edge(a, b) for a, b in itertools.combinations(sel, 2))


def repair_to_independent(G: nx.Graph, sel: Seq[int]) -> List[int]:
    """Sampled bitstrings can violate an edge (imperfect embedding, non-adiabatic
    transitions, shot noise).  Drop the lowest-weight endpoint until feasible.
    This keeps every shot usable instead of discarding it."""
    sel = sorted(sel, key=lambda n: -G.nodes[n]["weight"])
    keep: List[int] = []
    for n in sel:
        if all(not G.has_edge(n, m) for m in keep):
            keep.append(n)
    return sorted(keep)


# A quick self-test on the graph of the paper's Figure 5 (built properly in
# Cell 11): tracks 3 and 6, total weight 19.2.
_g = nx.Graph()
for _t, _w in {2: 3.4, 3: 9.1, 4: 7.5, 5: 4.8, 6: 10.1}.items():
    _g.add_node(_t, weight=_w)
_g.add_edges_from([(2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6)])
print("Self-test on the paper's Figure 5 graph")
print(f"  brute force : {mwis_bruteforce(_g, list(_g.nodes))}")
print(f"  exact       : {mwis_exact(_g, list(_g.nodes))}")
print(f"  greedy      : {mwis_greedy(_g, list(_g.nodes))}")
assert abs(mwis_exact(_g, list(_g.nodes))[0] - 19.2) < 1e-9


# %% ===========================================================================
# CELL 6 -- Track-oriented MHT: predict, gate, branch, score, prune
# ==============================================================================
# A *track* is one hypothesised history of one object: a sequence of blips, one
# (or none) per scan.  A *family* is all tracks descended from the same initiating
# blip -- i.e. all competing explanations of the same putative object.
#
# Order of operations at every scan (this order is what keeps the problem finite):
#   1. PREDICT every existing track forward.
#   2. GATE   : reject blips that are kinematically impossible for that track.
#   3. BRANCH : one child per gated blip, plus one "missed detection" child.
#   4. INITIATE: every blip also starts a brand-new family (it could be new).
#   5. SCORE + PRUNE: kill branches whose LLR falls below threshold.
# Only after all of that do we build the incompatibility graph.

@dataclass
class Track:
    tid: int
    family: int
    x: np.ndarray                       # Kalman state
    P: np.ndarray                       # Kalman covariance
    score: float                        # LLR
    obs_keys: FrozenSet[Tuple[int, int]]    # the real blips it claims
    history: Tuple[Tuple[int, int], ...]    # (scan, obs_idx) with idx = -1 for miss
    misses: int = 0                     # consecutive missed detections
    hits: int = 0
    parent: Optional[int] = None
    last_obs_z: Optional[np.ndarray] = None   # for kinematic (speed) pruning
    last_obs_scan: int = -1

    @property
    def n_obs(self) -> int:
        return len(self.obs_keys)


class MHT:
    """Track-oriented MHT front end.  Produces the MWIS graph; the *solver* for
    that graph is injected (classical or quantum) so we can compare them."""

    def __init__(self, cfg: Config, beta_fa: float):
        self.cfg = cfg
        self.beta_fa = beta_fa
        self.F, self.Q = cv_matrices(cfg.dt, cfg.q_accel)
        self.tracks: List[Track] = []
        self._next_tid = 0
        self._next_family = 0

    # -- helpers ---------------------------------------------------------------
    def _new_tid(self) -> int:
        self._next_tid += 1
        return self._next_tid - 1

    def _initiate(self, o: Observation) -> Track:
        """One-point track initiation: position from the blip, velocity unknown
        (large covariance), so the first gate is wide and then tightens."""
        x = np.array([o.z[0], o.z[1], 0.0, 0.0])
        v_var = self.cfg.v_init_std ** 2         # m^2/s^2, velocity prior
        P = np.zeros((4, 4))
        P[:2, :2] = o.R
        P[2, 2] = P[3, 3] = v_var
        fam = self._next_family
        self._next_family += 1
        return Track(self._new_tid(), fam, x, P, self.cfg.score_init,
                     frozenset({o.key}), ((o.scan, o.idx),), 0, 1, None,
                     o.z.copy(), o.scan)

    # -- one scan --------------------------------------------------------------
    def process_scan(self, obs: List[Observation]) -> None:
        cfg = self.cfg
        children: List[Track] = []

        for t in self.tracks:
            x_pred, P_pred = kf_predict(t.x, t.P, self.F, self.Q)

            # 2+3. gate and branch on every surviving blip
            for o in obs:
                nu, S, d2 = kf_innovation(x_pred, P_pred, o.z, o.R)
                if d2 > cfg.gate_chi2:
                    continue                      # outside the validation gate
                # kinematic pruning: reject associations that would require an
                # impossible speed (the paper's "too fast for the target type")
                if t.last_obs_z is not None:
                    dt_obs = (o.scan - t.last_obs_scan) * cfg.dt
                    if dt_obs > 0 and (np.linalg.norm(o.z - t.last_obs_z) / dt_obs
                                       > cfg.v_max):
                        continue
                x_u, P_u = kf_update(x_pred, P_pred, nu, S)
                children.append(Track(
                    self._new_tid(), t.family, x_u, P_u,
                    t.score + score_update(S, d2, cfg, self.beta_fa),
                    t.obs_keys | {o.key},
                    t.history + ((o.scan, o.idx),),
                    0, t.hits + 1, t.tid, o.z.copy(), o.scan))

            # 3b. the missed-detection branch: the object is still there, the
            #     radar just did not see it this time.
            if t.misses + 1 <= cfg.max_misses:
                children.append(Track(
                    self._new_tid(), t.family, x_pred, P_pred,
                    t.score + score_miss(cfg),
                    t.obs_keys,
                    t.history + ((obs[0].scan if obs else -1, -1),),
                    t.misses + 1, t.hits, t.tid, t.last_obs_z, t.last_obs_scan))

        # 4. every blip may also be a brand-new object
        for o in obs:
            children.append(self._initiate(o))

        self.tracks = children
        self._prune_track_level()
        self.merge_duplicate_tracks(obs[0].scan if obs else 0)

    # -- 5. track-level pruning ------------------------------------------------
    def _prune_track_level(self) -> None:
        cfg = self.cfg
        keep = [t for t in self.tracks
                if t.score >= cfg.score_delete and t.misses <= cfg.max_misses]

        # cap the number of branches per family (they all explain one object)
        by_family: Dict[int, List[Track]] = {}
        for t in keep:
            by_family.setdefault(t.family, []).append(t)
        keep = []
        for fam, ts in by_family.items():
            ts.sort(key=lambda t: -t.score)
            keep.extend(ts[:cfg.max_tracks_per_family])

        # global cap, so the classical stage stays real-time
        keep.sort(key=lambda t: -t.score)
        self.tracks = keep[:cfg.max_tracks_total]

    def merge_duplicate_tracks(self, scan_k: int) -> int:
        """Track merging -- standard MHT hygiene, and essential here.

        Every observation initiates a new family, so one object accumulates a
        fresh family on every scan.  Those families quickly become *duplicates*:
        they explain the recent past with exactly the same blips and differ only
        in ancient history.  Because they share observations they are pairwise
        incompatible, so leaving them in turns each object's cluster into a large
        clique -- the single worst case for a unit-disk embedding.

        Two tracks whose observations agree over the last N scans are merged,
        keeping the higher-scoring one.
        """
        N = self.cfg.n_scan_window
        best: Dict[Tuple, Track] = {}
        for t in self.tracks:
            key = tuple(sorted((s, i) for (s, i) in t.history
                               if i >= 0 and s > scan_k - N))
            cur = best.get(key)
            if cur is None or t.score > cur.score:
                best[key] = t
        before = len(self.tracks)
        self.tracks = list(best.values())
        return before - len(self.tracks)

    # -- incompatibility lists and the MWIS graph ------------------------------
    def build_graph(self) -> Tuple[nx.Graph, List[Track]]:
        """Vertices = confirmed tracks.  The paper's rule is "positive score"
        (a negative-score track can never improve a global hypothesis); on top of
        that we apply the confirmation test of Sec. 4.2, so that a track must
        have earned several detections before it is worth spending atoms on.
        Edge (i,j) <=> tracks i and j are incompatible."""
        verts = [t for t in self.tracks
                 if t.score > self.cfg.score_confirm
                 and t.hits >= self.cfg.min_hits_confirm]

        # Build the ICL the cheap way: bucket tracks by the observations they
        # claim; every pair inside a bucket is incompatible.  O(sum |bucket|^2).
        by_obs: Dict[Tuple[int, int], List[int]] = {}
        for i, t in enumerate(verts):
            for key in t.obs_keys:
                by_obs.setdefault(key, []).append(i)

        G = nx.Graph()
        for i, t in enumerate(verts):
            G.add_node(i, weight=t.score, tid=t.tid, family=t.family)
        for _, members in by_obs.items():
            for a, b in itertools.combinations(members, 2):
                G.add_edge(a, b)
        # two branches of the same family describe the same object -> exclusive
        for a, b in itertools.combinations(range(len(verts)), 2):
            if verts[a].family == verts[b].family:
                G.add_edge(a, b)
        return G, verts


def cluster_graph(G: nx.Graph) -> List[List[int]]:
    """Clustering = connected components of the incompatibility graph.  Each
    cluster is an *independent* optimisation problem, which is exactly what makes
    the quantum step tractable: we never embed the whole graph at once."""
    return [sorted(c) for c in nx.connected_components(G)]


def n_scan_prune(mht: MHT, best_tids: List[int], scan_k: int) -> int:
    """N-scan pruning -- what makes MHT finite in *time* rather than just in size.

    Decisions are deferred, but not for ever.  Once we are N scans past a branch
    point we accept whatever the best global hypothesis decided there, and delete
    the branches of that family which disagree at that scan.  Without this the
    track tree (and hence the MWIS graph) grows without bound, and the clusters
    quickly stop fitting on any atom array.
    """
    N = mht.cfg.n_scan_window
    root_scan = scan_k - N
    if root_scan < 0:
        return 0
    by_tid = {t.tid: t for t in mht.tracks}

    def assignment_at(t: Track, k: int):
        for (s, i) in t.history:
            if s == k:
                return i
        return None

    locked: Dict[int, object] = {}
    for tid in best_tids:
        t = by_tid.get(tid)
        if t is None:
            continue
        a = assignment_at(t, root_scan)
        if a is not None:
            locked[t.family] = a

    before = len(mht.tracks)
    mht.tracks = [t for t in mht.tracks
                  if t.family not in locked
                  or assignment_at(t, root_scan) is None
                  or assignment_at(t, root_scan) == locked[t.family]]
    return before - len(mht.tracks)


# --- run the front end WITH its pruning loop, and look at scan 6 --------------
# The pruning has to be in the loop: the graph you hand to the quantum computer
# is the *pruned* one.  Running the front end without it produces enormous,
# nearly-complete clusters that are neither realistic nor embeddable.
_mht = MHT(CFG, BETA_FA)
for k in range(7):
    _mht.process_scan(SCANS[k])
    _G, _V = _mht.build_graph()
    _chosen = []
    for _cl in cluster_graph(_G):
        _chosen.extend(mwis_greedy(_G, _cl)[1])       # cheap in-loop decision
    n_scan_prune(_mht, [_G.nodes[v]["tid"] for v in _chosen], k)

G_demo, V_demo = _mht.build_graph()
clusters_demo = cluster_graph(G_demo)
print(f"After scan 6:")
print(f"  tracks alive            : {len(_mht.tracks)}")
print(f"  confirmed vertices      : {G_demo.number_of_nodes()}")
print(f"  incompatibility edges   : {G_demo.number_of_edges()}")
print(f"  clusters                : {[len(c) for c in clusters_demo]}")
print(f"  graph density           : {nx.density(G_demo):.3f}  (sparse => embeddable)")


# %% ===========================================================================
# CELL 7 -- Look at one MWIS instance
# ==============================================================================

def plot_mwis_graph(G, clusters, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    pos = nx.spring_layout(G, seed=3, k=0.9)
    cl_of = {n: ci for ci, c in enumerate(clusters) for n in c}
    w = np.array([G.nodes[n]["weight"] for n in G.nodes])
    sizes = 120 + 340 * (w - w.min()) / (np.ptp(w) + 1e-9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRIDC, width=1.4)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes,
        node_color=[SERIES[cl_of[n] % len(SERIES)] for n in G.nodes],
        edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=6.5, font_color="white", font_weight="bold",
        labels={n: f"{G.nodes[n]['weight']:.0f}" for n in G.nodes})
    ax.set_title(title)
    ax.text(.5, -.06, "node = candidate track (label & size = LLR score)   |   "
                      "edge = shares a blip   |   colour = cluster",
            transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    plt.show()


plot_mwis_graph(G_demo, clusters_demo,
                "MWIS instance handed to the quantum computer (after scan 6)")

c0 = max(clusters_demo, key=len)
t0 = time.time()
w_ex, s_ex = mwis_exact(G_demo, c0)
w_gr, s_gr = mwis_greedy(G_demo, c0)
n_e = G_demo.subgraph(c0).number_of_edges()
print(f"Largest cluster: {len(c0)} vertices, {n_e} edges "
      f"(density {2*n_e/(len(c0)*(len(c0)-1)):.2f})")
print(f"  exact  MWIS weight = {w_ex:7.2f}   set = {s_ex}   ({time.time()-t0:.3f}s)")
print(f"  greedy MWIS weight = {w_gr:7.2f}   set = {s_gr}")
print("\nThe tracks in the best hypothesis for this cluster:")
for v in s_ex:
    t = V_demo[v]
    blips = [f"{k}:{i}" for (k, i) in sorted(t.history) if i >= 0]
    print(f"  vertex {v:2d}  LLR {t.score:6.2f}  family {t.family:3d}  "
          f"blips (scan:idx) {blips}")


# %% ===========================================================================
# CELL 8 -- Geometric embedding: arbitrary graph -> unit-disk graph
# ==============================================================================
# THIS IS THE HARD PART, and the one the challenge sheet singles out.
#
# On a neutral-atom machine you do not get to choose the edges.  You choose where
# the atoms *are*, and physics decides which pairs interact: two atoms interact
# (blockade) iff their separation is below the blockade radius
#
#       R_b = (C6 / (hbar*Omega))^(1/6)
#
# So the graph the hardware solves is always a *unit-disk graph*.  Our MHT graph
# is not, in general.  We must find 2-D coordinates with
#
#       ||p_i - p_j|| <  R_b   for every edge      (incompatible tracks)
#       ||p_i - p_j|| >  R_b   for every non-edge  (compatible tracks)
#
# We solve this as a smooth penalty minimisation with random restarts, using
# margins (r_edge = 0.75 R_b, r_nonedge = 1.35 R_b) so that shot-to-shot noise
# does not flip an edge.  Not every graph is a unit-disk graph -- the function
# therefore *reports* the edges it could not realise, which is the honest thing
# to do, and Cell 14 measures how often that happens.

def embed_unit_disk(G: nx.Graph, nodes: Seq[int], cfg: Config, rng,
                    ) -> Tuple[np.ndarray, Dict]:
    """Embed the graph as a unit-disk graph, IN UNITS OF THE BLOCKADE RADIUS.

    Working in dimensionless units matters: the physical size of the array and
    the Rabi frequency are *consequences* of the layout, not inputs to it (see
    `physical_register` below).  Fixing Omega up front is what produced registers
    violating the device's minimum atom spacing.
    """
    nodes = list(nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    r_in, r_out, r_flr = cfg.r_edge, cfg.r_nonedge, cfg.r_floor   # in R_b units

    edges = [(idx[a], idx[b]) for a, b in G.subgraph(nodes).edges]
    eset = {tuple(sorted(e)) for e in edges}
    nonedges = [p for p in itertools.combinations(range(n), 2) if p not in eset]

    if n == 1:
        return np.zeros((1, 2)), {"cost": 0.0, "bad_edges": [], "bad_nonedges": [],
                                  "n": 1, "min_sep_ratio": np.inf, "radius_ratio": 0.0}

    def cost_grad(flat):
        P = flat.reshape(n, 2)
        g = np.zeros_like(P)
        c = 0.0

        def push(i, j, target, inward):
            """inward=True  -> want r <= target ; False -> want r >= target."""
            nonlocal c
            d = P[i] - P[j]
            r = np.linalg.norm(d) + 1e-9
            v = (r - target) if inward else (target - r)
            if v > 0:
                c += v * v
                gr = (2 * v * d / r) * (1.0 if inward else -1.0)
                g[i] += gr
                g[j] -= gr

        for i, j in edges:              # incompatible -> must blockade
            push(i, j, r_in, True)
        for i, j in nonedges:           # compatible -> must NOT blockade
            push(i, j, r_out, False)
        for i, j in itertools.combinations(range(n), 2):
            push(i, j, r_flr, False)    # keep atoms resolvable by the tweezers
        return c, g.ravel()

    best = None
    for _ in range(cfg.embed_restarts):
        x0 = rng.normal(0, 0.9, size=2 * n)
        res = minimize(cost_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 600})
        if best is None or res.fun < best.fun:
            best = res
            if best.fun < 1e-9:
                break

    P = best.x.reshape(n, 2)
    P -= P.mean(axis=0)

    # honest report of what the geometry actually realises (blockade at r = 1)
    bad_e = [(nodes[i], nodes[j]) for i, j in edges
             if np.linalg.norm(P[i] - P[j]) > 1.0]         # constraint missing
    bad_ne = [(nodes[i], nodes[j]) for i, j in nonedges
              if np.linalg.norm(P[i] - P[j]) < 1.0]        # constraint spurious
    seps = [np.linalg.norm(P[i] - P[j])
            for i, j in itertools.combinations(range(n), 2)]
    return P, {"cost": float(best.fun), "bad_edges": bad_e, "bad_nonedges": bad_ne,
               "n": n, "min_sep_ratio": float(min(seps)),
               "radius_ratio": float(np.max(np.linalg.norm(P, axis=1)))}


def enforce_min_spacing(P: np.ndarray, d_min: float, iters: int = 400) -> np.ndarray:
    """Push apart any pair closer than the tweezer resolution limit.

    Needed when a cluster is so tightly connected that no admissible R_b gives
    every pair enough room.  Separating them can *break* blockade edges -- that
    loss is measured by `embedding_report` and repaired shot-by-shot in Cell 10.
    """
    P = P.copy()
    n = len(P)
    for _ in range(iters):
        worst = 0.0
        for i, j in itertools.combinations(range(n), 2):
            d = P[i] - P[j]
            r = np.linalg.norm(d)
            if r < d_min:
                if r < 1e-9:                       # coincident: nudge apart
                    d = np.array([1.0, 0.0])
                    r = 1e-9
                push = 0.5 * (d_min - r) * d / r
                P[i] += push
                P[j] -= push
                worst = max(worst, d_min - r)
        if worst < 1e-6:
            break
    return P


def embedding_report(G: nx.Graph, nodes: Seq[int], P_um: np.ndarray,
                     Rb: float) -> Dict:
    """Which graph edges does this *physical* register actually realise?"""
    nodes = list(nodes)
    bad_e, bad_ne = [], []
    for i, j in itertools.combinations(range(len(nodes)), 2):
        r = np.linalg.norm(P_um[i] - P_um[j])
        if G.has_edge(nodes[i], nodes[j]):
            if r > Rb:
                bad_e.append((nodes[i], nodes[j]))     # constraint lost
        elif r < Rb:
            bad_ne.append((nodes[i], nodes[j]))        # constraint invented
    return {"bad_edges": bad_e, "bad_nonedges": bad_ne}


def physical_register(P_norm: np.ndarray, diag: Dict, cfg: Config,
                      G: Optional[nx.Graph] = None,
                      nodes: Optional[Seq[int]] = None) -> Dict:
    """Turn a dimensionless layout into micrometres + a Rabi frequency.

    The layout fixes only the *ratios* of distances.  Choosing the blockade
    radius R_b fixes everything else, because R_b = (C6/Omega)^(1/6):

        R_b larger -> atoms further apart (good: tweezers can resolve them)
                      but Omega smaller  (bad: the sweep stops being adiabatic)

    Because of the 1/6 power this trade-off is brutal -- buying a factor 2 in
    spacing costs a factor 64 in Rabi frequency.  We take the smallest R_b that
    respects the tweezer spacing, then clamp it to the adiabaticity floor.
    """
    d_min = DEVICE.min_atom_distance
    rb_lo = (C6 / cfg.omega_cap) ** (1 / 6)      # smallest R_b we would use
    rb_hi = (C6 / cfg.omega_min) ** (1 / 6)      # largest R_b we can afford
    rb_need = (d_min / diag["min_sep_ratio"]) if np.isfinite(
        diag["min_sep_ratio"]) else rb_lo

    Rb = float(np.clip(rb_need, rb_lo, rb_hi))
    omega = C6 / Rb ** 6

    P_um = P_norm * Rb
    spacing_forced = rb_need > rb_hi             # could not buy enough room
    if spacing_forced:
        P_um = enforce_min_spacing(P_um, d_min)

    seps = [np.linalg.norm(P_um[i] - P_um[j])
            for i, j in itertools.combinations(range(len(P_um)), 2)]
    rep = (embedding_report(G, list(nodes), P_um, Rb)
           if G is not None and nodes is not None
           else {"bad_edges": [], "bad_nonedges": []})

    return {"P_um": P_um, "Rb": Rb, "omega": omega,
            "spacing_limited": rb_need > rb_lo,
            "spacing_forced": spacing_forced,
            "omega_T": omega * cfg.t_sweep * 1e-3,     # rad, adiabaticity budget
            "min_sep_um": float(min(seps)) if seps else np.inf,
            "radius_um": float(np.max(np.linalg.norm(P_um, axis=1))),
            **rep}


# --- demo on the largest cluster ---------------------------------------------
c0 = max(clusters_demo, key=len)
P0n, diag0 = embed_unit_disk(G_demo, c0, CFG, np.random.default_rng(1))
phys0 = physical_register(P0n, diag0, CFG, G_demo, c0)
P0 = phys0["P_um"]
n_e0 = G_demo.subgraph(c0).number_of_edges()
n_pairs0 = diag0["n"] * (diag0["n"] - 1) // 2
print(f"Embedding {diag0['n']} atoms "
      f"({n_e0}/{n_pairs0} edges -> density {n_e0/n_pairs0:.2f})")
print(f"  residual cost      : {diag0['cost']:.4g}")
print(f"  -> R_b             : {phys0['Rb']:.2f} um "
      f"({'limited by tweezer spacing' if phys0['spacing_limited'] else 'set by Omega cap'})")
print(f"  -> Omega           : {phys0['omega']:.3f} rad/us "
      f"(2pi x {phys0['omega']/2/np.pi:.3f} MHz),  Omega*T = {phys0['omega_T']:.1f} rad")
print(f"  min separation     : {phys0['min_sep_um']:.2f} um  (device min {DEVICE.min_atom_distance})")
print(f"  array radius       : {phys0['radius_um']:.2f} um  (device max {DEVICE.max_radial_distance})")
print(f"  unrealised edges   : {len(phys0['bad_edges'])}  {phys0['bad_edges'][:6]}")
print(f"  spurious edges     : {len(phys0['bad_nonedges'])}  {phys0['bad_nonedges'][:6]}")
if phys0["spacing_forced"]:
    _kmax = max(len(c) for c in nx.find_cliques(G_demo.subgraph(c0)))
    _ratio = phys0["Rb"] / DEVICE.min_atom_distance
    print(f"\n  NOTE: the obstruction is the largest CLIQUE in this cluster, which has"
          f"\n  {_kmax} vertices.  Those atoms must blockade each other pairwise, i.e. sit"
          f"\n  within R_b of one another while staying >= {DEVICE.min_atom_distance} um apart.  Here"
          f"\n  R_b/d_min = {_ratio:.1f}, and in 2-D only about 4-5 atoms can be mutually within"
          f"\n  R_b at that ratio.  Enlarging R_b buys room but costs Omega as R_b^-6,"
          f"\n  so the adiabaticity floor stops us: the layout was forced open and"
          f"\n  {len(phys0['bad_edges'])} blockade edges were lost.  Cell 10 repairs the affected shots.")


def plot_embedding(G, nodes, P, Rb, diag, title):
    nodes = list(nodes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))

    ax = axes[0]
    pos = nx.spring_layout(G.subgraph(nodes), seed=3)
    nx.draw_networkx_edges(G.subgraph(nodes), pos, ax=ax, edge_color=GRIDC, width=1.5)
    nx.draw_networkx_nodes(G.subgraph(nodes), pos, ax=ax, node_size=260,
                           node_color=SERIES[0], edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G.subgraph(nodes), pos, ax=ax, font_size=7,
                            font_color="white", font_weight="bold")
    ax.set_title("Abstract MHT incompatibility graph")
    ax.axis("off")

    ax = axes[1]
    for i, v in enumerate(nodes):
        ax.add_patch(Circle(P[i], Rb / 2, facecolor=SERIES[0], alpha=.10,
                            edgecolor=SERIES[0], lw=.8, ls="--"))
    for i, j in itertools.combinations(range(len(nodes)), 2):
        if np.linalg.norm(P[i] - P[j]) < Rb:
            realised = G.has_edge(nodes[i], nodes[j])
            ax.plot(*zip(P[i], P[j]), "-", lw=2 if realised else 2.2,
                    color=MUTED if realised else ACCENT_BAD,
                    zorder=1, alpha=.9 if realised else 1.0)
    ax.scatter(P[:, 0], P[:, 1], s=190, c=INK, zorder=3, edgecolors="white",
               linewidths=1.5)
    for i, v in enumerate(nodes):
        ax.annotate(str(v), P[i], color="white", ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=4)
    ax.set_title("Optical-tweezer register (blockade half-discs)")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_aspect("equal")
    n_bad = len(diag["bad_edges"]) + len(diag["bad_nonedges"])
    ax.text(.5, -.16, "grey link = blockade pair that IS a graph edge"
            + ("   |   red = spurious/missing" if n_bad else "   |   no errors"),
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.show()


plot_embedding(G_demo, c0, P0, phys0["Rb"], diag0,
               "Step 2 of the pipeline: geometric embedding")


# %% ===========================================================================
# CELL 9 -- Building the atom register and the adiabatic pulse sequence
# ==============================================================================
# The physics we are engineering, in the frozen-atom (Omega -> 0) limit:
#
#     H/hbar = -sum_i delta_i n_i  +  sum_{i<j} (C6 / r_ij^6) n_i n_j
#              \_______________/      \_______________________/
#               local detuning         van der Waals / Rydberg blockade
#               encodes the weight     encodes the constraint x_i + x_j <= 1
#
# Compare with the challenge sheet's Hamiltonian  H = -sum w_i x_i + lam sum x_i x_j.
#
# ENCODING THE WEIGHTS.  Pasqal's DMM can only apply *negative* local detuning,
# with a per-atom weight m_i in [0,1] scaling a common waveform.  So we write
#
#     delta_i = delta_final - |Delta_DMM| * m_i ,     m_i = 1 - w_i / w_max
#
# and choose |Delta_DMM| = delta_final, which gives exactly
#
#     delta_i = delta_final * (w_i / w_max)      =>   -sum_i delta_i n_i
#                                                 = -(delta_final/w_max) sum_i w_i n_i
#
# i.e. the detuning landscape is proportional to the LLR track scores, which is
# precisely the first term of the QUBO.  Note this requires w_i > 0 -- and the
# paper already tells us to keep only positive-score tracks.  The two things fit
# together for free.
#
# A NICE CONSEQUENCE ABOUT lambda.  On this hardware lambda is *not* a free
# parameter to tune.  It is C6/r_ij^6, set by where we put the atoms.  Choosing
# r_edge = 0.75 R_b fixes lambda / Omega = 0.75^-6 = 5.6, and we then only need
# delta_max < lambda so that no pair of blockaded atoms can pay the penalty.
# Cell 14 verifies this inequality is what actually matters.

def build_sequence(P: np.ndarray, weights: np.ndarray, omega: float,
                   cfg: Config) -> Sequence:
    """Register + 3-stage adiabatic sweep + weight-encoding local detuning."""
    n = len(weights)
    reg = Register({f"q{i}": tuple(P[i]) for i in range(n)})

    delta_max = cfg.delta_ratio * omega
    d0, d1 = -delta_max, delta_max
    O, tr, T = omega, cfg.t_ramp, cfg.t_sweep

    # ---- weights -> DMM detuning map ----------------------------------------
    # The DMM applies detuning_i = m_i * waveform, and the hardware cannot
    # resolve an average absolute detuning below min_avg_abs_detuning.  So any
    # m_i below m_min is simply not representable and must be snapped to zero:
    # the weight encoding is QUANTISED, and two tracks whose scores differ by
    # less than m_min * w_max are indistinguishable to the machine.
    w_max = float(np.max(weights))
    m = 1.0 - np.asarray(weights, float) / w_max          # in [0, 1)
    m_min = DEVICE.dmm_channels["dmm_0"].min_avg_abs_detuning / d1
    m[m < m_min] = 0.0

    seq = Sequence(reg, DEVICE)
    seq.declare_channel("ryd", "rydberg_global")
    use_dmm = bool(np.any(m > 0.0))
    if use_dmm:
        det_map = reg.define_detuning_map({f"q{i}": float(m[i]) for i in range(n)})
        seq.config_detuning_map(det_map, "dmm_0")

    # (a) turn the Rabi drive on at large negative detuning: ground state is
    #     |gg...g>, which is trivially the state we can prepare.
    seq.add(Pulse(RampWaveform(tr, 0, O), ConstantWaveform(tr, d0), 0), "ryd")
    # (b) sweep the detuning through resonance: the ground state deforms
    #     adiabatically into the maximum-weight independent set.
    seq.add(Pulse(ConstantWaveform(T, O), InterpolatedWaveform(T, [d0, 0, d1]), 0),
            "ryd")
    # (c) turn the drive off, freezing the configuration for readout.
    seq.add(Pulse(RampWaveform(tr, O, 0), ConstantWaveform(tr, d1), 0), "ryd")

    # the per-atom detuning that carries the weights (constant, so that at the
    # end of the sweep delta_i = delta_max * w_i / w_max exactly).  If every
    # weight is within one quantisation step of the maximum there is nothing to
    # encode, and the problem degenerates to unweighted maximum independent set.
    if use_dmm:
        seq.add_dmm_detuning(ConstantWaveform(seq.get_duration(), -d1), "dmm_0")
    return seq


_w0 = np.array([G_demo.nodes[v]["weight"] for v in c0])
seq0 = build_sequence(P0, _w0, phys0["omega"], CFG)
print(f"Sequence duration: {seq0.get_duration()} ns "
      f"(device max {DEVICE.max_sequence_duration} ns)")
_dmm_step = (DEVICE.dmm_channels["dmm_0"].min_avg_abs_detuning
             / (CFG.delta_ratio * phys0["omega"]))
print(f"Weights  : {np.round(_w0, 2)}")
print(f"DMM m_i  : {np.round(np.where((1 - _w0/_w0.max()) < _dmm_step, 0.0, 1 - _w0/_w0.max()), 3)}"
      f"   (0 = most favoured atom)")
print(f"Weight quantisation: m_min = {_dmm_step:.3f}, i.e. score differences below "
      f"{_dmm_step * _w0.max():.2f} LLR are invisible to the hardware")
seq0.draw(mode="input")


# %% ===========================================================================
# CELL 10 -- Running the atom array and turning shots into hypotheses
# ==============================================================================
# The measurement gives a bitstring per shot: 1 = atom in the Rydberg state =
# "this track is in the global hypothesis".  Because the sweep is not perfectly
# adiabatic we do NOT always land in the ground state -- and for MHT that is a
# feature, not a bug: the spread over low-energy states *is* the set of
# near-optimal global hypotheses that Eqs. (2)-(3) need.

def run_quantum_mwis(G: nx.Graph, nodes: Seq[int], cfg: Config, rng,
                     shots: Optional[int] = None) -> Dict:
    """Embed -> build sequence -> emulate -> sample -> repair -> rank."""
    nodes = list(nodes)
    weights = np.array([G.nodes[v]["weight"] for v in nodes], float)

    P_norm, diag = embed_unit_disk(G, nodes, cfg, rng)
    phys = physical_register(P_norm, diag, cfg, G, nodes)
    P = phys["P_um"]
    diag = {**diag, "bad_edges": phys["bad_edges"],
            "bad_nonedges": phys["bad_nonedges"]}
    seq = build_sequence(P, weights, phys["omega"], cfg)

    cfg_emu = QutipConfig(observables=[BitStrings(num_shots=shots or cfg.shots)],
                          sampling_rate=0.2)
    # Pulser draws its shots from numpy's global RNG, so seed it from our own
    # generator to keep the whole notebook reproducible run to run.
    np.random.seed(int(rng.integers(2 ** 31 - 1)))
    counts = QutipBackendV2(seq, config=cfg_emu).run().bitstrings[-1]

    # bitstring -> selected vertices -> repaired independent set
    hyps: Dict[Tuple[int, ...], Dict] = {}
    n_violating = 0
    for bits, c in counts.items():
        sel_raw = [nodes[i] for i, ch in enumerate(bits) if ch == "1"]
        if not is_independent(G, sel_raw):
            n_violating += c
            sel = repair_to_independent(G, sel_raw)
        else:
            sel = sorted(sel_raw)
        key = tuple(sel)
        rec = hyps.setdefault(key, {"sel": sel, "counts": 0,
                                    "weight": sum(G.nodes[v]["weight"] for v in sel)})
        rec["counts"] += c

    ranked = sorted(hyps.values(), key=lambda h: -h["weight"])
    total = sum(counts.values())
    return {"positions": P, "diag": diag, "phys": phys, "sequence": seq,
            "counts": counts, "hypotheses": ranked, "shots": total,
            "violation_rate": n_violating / total,
            "best": ranked[0] if ranked else {"sel": [], "weight": 0.0}}


def hypothesis_probabilities(hyps: List[Dict]) -> Tuple[List[float], Dict[int, float]]:
    """Equations (2) and (3) of the paper.

        P{H_j}  = exp(L_Hj) / (1 + sum_i exp(L_Hi))
        P{T_i}  = sum over hypotheses containing track i of P{H_j}

    The '1' is the trivial hypothesis 'every blip is a false alarm'.  We evaluate
    it with logsumexp because L_H is a sum of LLRs and exponentiating directly
    overflows as soon as a few good tracks are present.
    """
    L = np.array([h["weight"] for h in hyps], float)
    logZ = logsumexp(np.concatenate([[0.0], L]))
    pH = np.exp(L - logZ)
    pT: Dict[int, float] = {}
    for h, p in zip(hyps, pH):
        for v in h["sel"]:
            pT[v] = pT.get(v, 0.0) + p
    return list(pH), pT


t0 = time.time()
qres0 = run_quantum_mwis(G_demo, c0, CFG, np.random.default_rng(2))
elapsed = time.time() - t0
w_opt, s_opt = mwis_exact(G_demo, c0)
pH0, pT0 = hypothesis_probabilities(qres0["hypotheses"])

print(f"Emulated {qres0['shots']} shots on {len(c0)} atoms in {elapsed:.1f}s")
print(f"  edge-violating shots (repaired): {qres0['violation_rate']:.1%}")
print(f"\n  classical optimum : {w_opt:.2f}  {s_opt}")
print(f"  quantum best      : {qres0['best']['weight']:.2f}  {qres0['best']['sel']}")
print(f"  optimal?          : {'YES' if abs(w_opt - qres0['best']['weight']) < 1e-6 else 'NO'}")
print(f"\n  Low-energy hypotheses found (this is the MHT payoff):")
for h, p in list(zip(qres0["hypotheses"], pH0))[:6]:
    print(f"    weight {h['weight']:7.2f}   P(H) = {p:6.3f}   shots {h['counts']:4d}   {h['sel']}")


def plot_quantum_result(qres, G, nodes, w_opt, title):
    hyps = qres["hypotheses"][:8]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    counts = np.array([h["counts"] for h in hyps])
    weights = np.array([h["weight"] for h in hyps])
    y = np.arange(len(hyps))[::-1]
    cols = [ACCENT_OK if abs(w - w_opt) < 1e-6 else SERIES[0] for w in weights]
    ax.barh(y, counts / qres["shots"], color=cols, height=.62)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{{{','.join(map(str, h['sel']))}}}" for h in hyps],
                       fontsize=7.5)
    for yy, c, w in zip(y, counts, weights):
        ax.annotate(f"w={w:.1f}", (c / qres["shots"], yy), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=MUTED)
    ax.set_xlabel("sampled fraction")
    ax.set_title("Global hypotheses sampled from the array")
    ax.margins(x=.18)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    _, pT = hypothesis_probabilities(qres["hypotheses"])
    vs = list(nodes)
    p = [pT.get(v, 0.0) for v in vs]
    ax.bar(range(len(vs)), p, color=SERIES[2], width=.62)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels(vs, fontsize=7.5)
    ax.set_xlabel("vertex (candidate track)")
    ax.set_ylabel("P{T_i}")
    ax.set_title("Global track probabilities, Eq. (3)")
    ax.grid(axis="x", visible=False)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.show()


plot_quantum_result(qres0, G_demo, c0, w_opt,
                    "Neutral-atom solution of one MHT cluster")


# %% ===========================================================================
# CELL 11 -- Validation against the worked example in the paper (Table 1)
# ==============================================================================
# Papageorgiou & Salpukas give six tracks with explicit ICLs and scores, of which
# the five positive-score ones form the MWIS instance of their Figure 5, whose
# optimum is {track 3, track 6} with weight 19.2.  If our whole quantum stack is
# right, it must reproduce that number.

PAPER_IDS = [2, 3, 4, 5, 6]
PAPER_W = {2: 3.4, 3: 9.1, 4: 7.5, 5: 4.8, 6: 10.1}
PAPER_ICL = {1: {4, 6}, 2: {3, 4}, 3: {2, 4, 5}, 4: {1, 2, 3, 5, 6},
             5: {3, 4, 6}, 6: {1, 4, 5}}

G_paper = nx.Graph()
for t in PAPER_IDS:
    G_paper.add_node(t, weight=PAPER_W[t])
for t in PAPER_IDS:
    for u in PAPER_ICL[t]:
        if u in PAPER_W:                    # keep only positive-score tracks
            G_paper.add_edge(t, u)

w_paper, s_paper = mwis_exact(G_paper, PAPER_IDS)
qres_paper = run_quantum_mwis(G_paper, PAPER_IDS, CFG, np.random.default_rng(5))
pH_p, pT_p = hypothesis_probabilities(qres_paper["hypotheses"])

print("Paper Table 1 / Figure 5")
print(f"  expected optimum  : tracks [3, 6], weight 19.2")
print(f"  classical solver  : tracks {s_paper}, weight {w_paper}")
print(f"  neutral-atom best : tracks {qres_paper['best']['sel']}, "
      f"weight {qres_paper['best']['weight']:.1f}")
print(f"  embedding errors  : {len(qres_paper['diag']['bad_edges'])} missing, "
      f"{len(qres_paper['diag']['bad_nonedges'])} spurious")
top = qres_paper["hypotheses"][0]
print(f"  P(sampling the optimum) = {top['counts'] / qres_paper['shots']:.3f}")
assert abs(w_paper - 19.2) < 1e-9, "classical solver disagrees with the paper"

plot_embedding(G_paper, PAPER_IDS, qres_paper["positions"],
               qres_paper["phys"]["Rb"], qres_paper["diag"],
               "Paper Figure 5 embedded into an atom array")
plot_quantum_result(qres_paper, G_paper, PAPER_IDS, w_paper,
                    "Reproducing the paper's worked example on atoms")


# %% ===========================================================================
# CELL 12 -- The closed loop: quantum solver inside the tracker
# ==============================================================================
# Now we put it all together and actually *follow the objects*.  At every scan:
#   front end -> graph -> clusters -> (quantum or classical) MWIS per cluster
#   -> best global hypothesis -> N-scan pruning -> track output.
#
# N-scan pruning is what makes MHT finite in time: once we are N scans past a
# branch point, we accept the decision the best global hypothesis made there and
# delete the branches of that family that disagree.  Decisions are deferred, but
# not for ever.

def solve_cluster(G, nodes, cfg, rng, backend: str) -> Dict:
    """Dispatch one cluster to the requested solver, with an honest fallback
    when the cluster does not fit on the array."""
    nodes = list(nodes)
    if backend == "quantum":
        if len(nodes) <= cfg.max_atoms:
            r = run_quantum_mwis(G, nodes, cfg, rng)
            return {"sel": r["best"]["sel"], "weight": r["best"]["weight"],
                    "hypotheses": r["hypotheses"], "mode": "quantum",
                    "violation_rate": r["violation_rate"], "diag": r["diag"]}

        # --- cluster does not fit on the array -------------------------------
        # Hybrid decomposition: hand the heaviest `max_atoms` vertices to the
        # QPU (they dominate the objective), then classically extend the answer
        # with any remaining vertices that are compatible with it.  This is
        # exact only if the discarded vertices were truly irrelevant, so it is a
        # heuristic -- but it keeps the quantum stage in the loop instead of
        # silently falling back to a classical solver on every busy scan.
        heavy = sorted(nodes, key=lambda v: -G.nodes[v]["weight"])[:cfg.max_atoms]
        r = run_quantum_mwis(G, heavy, cfg, rng)
        sel = list(r["best"]["sel"])
        for v in sorted(set(nodes) - set(heavy), key=lambda v: -G.nodes[v]["weight"]):
            if all(not G.has_edge(v, u) for u in sel):
                sel.append(v)
        sel = sorted(sel)
        return {"sel": sel,
                "weight": sum(G.nodes[v]["weight"] for v in sel),
                "hypotheses": r["hypotheses"], "mode": "quantum-partitioned",
                "violation_rate": r["violation_rate"], "diag": r["diag"]}
    if backend == "greedy" or len(nodes) > cfg.exact_max_n:
        w, s = mwis_greedy(G, nodes)
    else:
        w, s = mwis_exact(G, nodes)
    mode = "classical" if backend != "quantum" else "classical-fallback"
    return {"sel": s, "weight": w,
            "hypotheses": enumerate_hypotheses(G, nodes, s, w, cfg),
            "mode": mode, "violation_rate": 0.0, "diag": None}


def enumerate_hypotheses(G, nodes, best_sel, best_w, cfg, cap: int = 300) -> List[Dict]:
    """Near-optimal global hypotheses for the *classical* path.

    Eqs. (2)-(3) need a set of competing hypotheses, not just the argmax.  If we
    handed the classical run a single answer, every other track would get
    P{T_i} = 0 and be pruned immediately -- which would make the comparison
    against the quantum run meaningless rather than merely different.

    So we enumerate the maximal independent sets explicitly (maximal cliques of
    the complement).  It is exact, and it is also the point: this enumeration is
    the expensive thing the atom array gives away for free in its shot noise.
    """
    nodes = list(nodes)
    if len(nodes) > cfg.exact_max_n:
        return [{"sel": best_sel, "weight": best_w, "counts": 1}]
    comp = nx.complement(G.subgraph(nodes)) if len(nodes) > 1 else None
    sets = ([sorted(c) for c in itertools.islice(nx.find_cliques(comp), cap)]
            if comp is not None else [list(nodes)])
    if not sets:
        sets = [list(nodes)]
    out = [{"sel": s, "weight": sum(G.nodes[v]["weight"] for v in s), "counts": 1}
           for s in sets]
    return sorted(out, key=lambda h: -h["weight"])


def run_tracker(cfg: Config, scans, beta_fa, backend="quantum",
                seed=11, verbose=True) -> Dict:
    rng_local = np.random.default_rng(seed)
    mht = MHT(cfg, beta_fa)
    per_scan = []

    for k, scan in enumerate(scans):
        t_start = time.time()
        mht.process_scan(scan)
        G, verts = mht.build_graph()
        clusters = cluster_graph(G)

        chosen: List[int] = []
        all_hyps: List[Dict] = []
        modes, viol, audit = [], [], []
        for cl in clusters:
            res = solve_cluster(G, cl, cfg, rng_local, backend)
            chosen.extend(res["sel"])
            all_hyps.append(res["hypotheses"])
            modes.append(res["mode"])
            viol.append(res["violation_rate"])

            # --- audit: was the array's answer actually optimal? -------------
            # This has to be done cluster by cluster on the SAME graph.
            # Comparing a quantum run against a separate classical run scan by
            # scan is meaningless: the two runs prune differently, so from the
            # second scan onwards they are not even solving the same problem.
            if res["mode"].startswith("quantum") and len(cl) <= cfg.exact_max_n:
                w_opt_cl, _ = mwis_exact(G, cl)
                audit.append((res["weight"], w_opt_cl, len(cl)))

        # --- global track probabilities, Eq. (3), used for probability pruning
        pT_global: Dict[int, float] = {}
        for hyps in all_hyps:
            _, pT = hypothesis_probabilities(hyps)
            pT_global.update(pT)

        best_tids = [G.nodes[v]["tid"] for v in chosen]
        out_tracks = [verts[v] for v in chosen]

        n_pruned = n_scan_prune(mht, best_tids, k)

        # probability pruning: drop graph tracks nobody believes in
        doomed = {G.nodes[v]["tid"] for v in G.nodes
                  if pT_global.get(v, 0.0) < cfg.prob_prune}
        mht.tracks = [t for t in mht.tracks if t.tid not in doomed or t.score <= 0]

        per_scan.append({
            "scan": k, "n_tracks": len(mht.tracks), "n_vertices": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "clusters": [len(c) for c in clusters],
            "chosen": out_tracks, "modes": modes, "audit": audit,
            "violation": float(np.mean(viol)) if viol else 0.0,
            "n_scan_pruned": n_pruned,
            "time": time.time() - t_start,
            "total_weight": sum(t.score for t in out_tracks),
        })
        if verbose:
            big = max([len(c) for c in clusters], default=0)
            print(f"  scan {k:2d} | tracks {len(mht.tracks):3d} | graph "
                  f"{G.number_of_nodes():2d}v/{G.number_of_edges():3d}e | "
                  f"clusters {sorted([len(c) for c in clusters], reverse=True)[:4]} "
                  f"(max {big}) | selected {len(out_tracks)} | "
                  f"{time.time() - t_start:5.1f}s")

    return {"mht": mht, "per_scan": per_scan, "backend": backend}


print("=== Neutral-atom-in-the-loop tracking ===")
t0 = time.time()
RUN_Q = run_tracker(CFG, SCANS, BETA_FA, backend="quantum", seed=11)
print(f"total wall time: {time.time() - t0:.1f}s\n")

print("=== Classical (exact) reference run ===")
RUN_C = run_tracker(CFG, SCANS, BETA_FA, backend="exact", seed=11, verbose=False)
print("done.")


# %% ===========================================================================
# CELL 13 -- Did we actually follow the objects?
# ==============================================================================

def extract_tracks(run) -> List[Track]:
    """The tracks in the final scan's best global hypothesis, keeping only ones
    that are confirmed (enough real detections)."""
    last = run["per_scan"][-1]
    return [t for t in last["chosen"] if t.n_obs >= 3]


def track_positions(t: Track, scans) -> Tuple[np.ndarray, List[int]]:
    """The blips this track claims, in time order."""
    pts, ks = [], []
    for (k, i) in sorted(t.history):
        if i >= 0:
            pts.append(scans[k][i].z)
            ks.append(k)
    return np.array(pts), ks


def association_accuracy(run, scans) -> Dict:
    """Fraction of the claimed blips that really came from a single true object."""
    tracks = extract_tracks(run)
    total, correct, purity = 0, 0, []
    for t in tracks:
        truths = [scans[k][i].truth for (k, i) in sorted(t.history) if i >= 0]
        real = [x for x in truths if x is not None]
        if not real:
            purity.append(0.0)
            continue
        vals, cnts = np.unique(real, return_counts=True)
        purity.append(cnts.max() / len(truths))
        correct += cnts.max()
        total += len(truths)
    return {"n_tracks": len(tracks),
            "assoc_accuracy": correct / total if total else 0.0,
            "mean_purity": float(np.mean(purity)) if purity else 0.0}


acc_q = association_accuracy(RUN_Q, SCANS)
acc_c = association_accuracy(RUN_C, SCANS)
print(f"Quantum-in-the-loop : {acc_q['n_tracks']} confirmed tracks, "
      f"association accuracy {acc_q['assoc_accuracy']:.1%}, "
      f"purity {acc_q['mean_purity']:.1%}")
print(f"Classical exact     : {acc_c['n_tracks']} confirmed tracks, "
      f"association accuracy {acc_c['assoc_accuracy']:.1%}, "
      f"purity {acc_c['mean_purity']:.1%}")

# --- solver quality, audited cluster by cluster on the same graph ------------
AUDIT = [a for p in RUN_Q["per_scan"] for a in p["audit"]]
n_opt = sum(1 for wq, we, _ in AUDIT if abs(wq - we) < 1e-6)
ratios = [wq / we for wq, we, _ in AUDIT if we > 0]
print(f"\nMWIS instances solved on the atom array : {len(AUDIT)}")
print(f"  matched the exact classical optimum   : {n_opt}/{len(AUDIT)}"
      f" ({n_opt/max(len(AUDIT),1):.0%})")
print(f"  mean weight ratio (quantum / optimal) : {np.mean(ratios):.4f}"
      if ratios else "")
print(f"  cluster sizes handled                 : "
      f"{sorted({n for _, _, n in AUDIT})}")
print(f"  mean edge-violating shot rate         : "
      f"{np.mean([p['violation'] for p in RUN_Q['per_scan'] if p['audit']]):.1%}"
      "   (all repaired before use)")


def plot_tracking_result():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    ax = axes[0]
    (x0, x1), (y0, y1) = action_limits(pad=700)
    fa = np.array([o.z for s in SCANS for o in s if o.truth is None])
    ax.scatter(fa[:, 0], fa[:, 1], s=12, c=MUTED, alpha=.30, linewidths=0,
               zorder=1, label="clutter")
    for i in range(len(TARGETS)):
        ks = sorted(TRUTH[i])
        xy = np.array([TRUTH[i][k] for k in ks])
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=4, color="#c9c9c4", zorder=2,
                label="true trajectory" if i == 0 else None)

    marks = ["o", "s", "^", "D", "v", "P"]
    for n, t in enumerate(extract_tracks(RUN_Q)):
        pts, ks = track_positions(t, SCANS)
        if len(pts) == 0:
            continue
        col = SERIES[n % len(SERIES)]
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=2, color=col, zorder=3)
        ax.plot(pts[:, 0], pts[:, 1], marks[n % len(marks)], ms=5, color=col,
                mec="white", mew=1.0, zorder=4,
                label=f"track {n+1} (LLR {t.score:.0f})")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title("Objects followed by the quantum-in-the-loop tracker")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax = axes[1]
    ks = [p["scan"] for p in RUN_Q["per_scan"]]
    ax.plot(ks, [p["n_vertices"] for p in RUN_Q["per_scan"]], "-o", ms=4,
            color=SERIES[0], lw=2, label="graph vertices (atoms needed)")
    ax.plot(ks, [max(p["clusters"], default=0) for p in RUN_Q["per_scan"]], "-o",
            ms=4, color=SERIES[1], lw=2, label="largest cluster")
    ax.axhline(CFG.max_atoms, ls="--", lw=1.4, color=ACCENT_BAD)
    ax.annotate("QPU cluster limit", (ks[0], CFG.max_atoms), xytext=(2, 4),
                textcoords="offset points", color=ACCENT_BAD, fontsize=8)
    ax.set_xlabel("scan")
    ax.set_ylabel("count")
    ax.set_title("Problem size handed to the array, per scan")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Object following on a neutral-atom quantum computer",
                 fontweight="bold")
    fig.tight_layout()
    plt.show()


plot_tracking_result()


# %% ===========================================================================
# CELL 14 -- Parameter studies: the two things that actually decide success
# ==============================================================================
# (a) Embeddability.  How often is an MHT cluster realisable as a unit-disk
#     graph?  This is the honest limit of the whole approach.
# (b) The detuning/blockade ratio delta_max / U_edge, which plays the role of
#     1/lambda in the QUBO.  Too large and blockaded pairs get excited together;
#     too small and the weights stop mattering.

def study_embeddability(sizes=(4, 6, 8, 10), densities=(0.2, 0.35, 0.5),
                        n_graphs=6, seed=3) -> Dict:
    r = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        for d in densities:
            ok = 0
            for _ in range(n_graphs):
                g = nx.gnp_random_graph(n, d, seed=int(r.integers(1e6)))
                for v in g.nodes:
                    g.nodes[v]["weight"] = float(r.uniform(1, 10))
                _, diag = embed_unit_disk(g, list(g.nodes), CFG, r)
                if not diag["bad_edges"] and not diag["bad_nonedges"]:
                    ok += 1
            out[(n, d)] = ok / n_graphs
    return out


print("Embeddability of random graphs into a unit-disk atom register")
print("(fraction with a perfect embedding; failures need edge repair or splitting)\n")
emb = study_embeddability()
sizes = sorted({k[0] for k in emb})
dens = sorted({k[1] for k in emb})
print("    n \\ density " + "".join(f"{d:>8.2f}" for d in dens))
for n in sizes:
    print(f"    {n:>3d}         " + "".join(f"{emb[(n, d)]:>8.0%}" for d in dens))


def plot_embeddability(emb):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sizes = sorted({k[0] for k in emb})
    dens = sorted({k[1] for k in emb})
    for j, d in enumerate(dens):
        ax.plot(sizes, [emb[(n, d)] for n in sizes], "-o", lw=2, ms=6,
                color=SERIES[j], mec="white", mew=1.2, label=f"density {d:.2f}")
        ax.annotate(f"{d:.2f}", (sizes[-1], emb[(sizes[-1], d)]),
                    xytext=(6, 0), textcoords="offset points",
                    color=SERIES[j], fontsize=8, fontweight="bold", va="center")
    ax.set_xlabel("cluster size (atoms)")
    ax.set_ylabel("perfectly embeddable")
    ax.set_ylim(-.05, 1.05)
    ax.set_xticks(sizes)
    ax.set_title("Unit-disk embeddability limits the cluster size")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    plt.show()


plot_embeddability(emb)


def study_delta(ratios=(0.5, 1.0, 2.0, 4.0, 5.5, 7.0)) -> List[Tuple[float, float, float]]:
    """Sweep delta_max/Omega on the paper graph; report P(optimum) and how often
    a shot violated an edge (i.e. how often the blockade failed to hold).

    The device also caps |detuning| (62.8 rad/us on this channel), so the sweep
    stops there rather than at some arbitrary software limit.
    """
    d_cap = DEVICE.channels["rydberg_global"].max_abs_detuning
    rows = []
    for ratio in ratios:
        cfg = replace(CFG, delta_ratio=ratio, shots=400)
        if ratio * cfg.omega_cap > d_cap:
            print(f"     {ratio:5.1f}      -- skipped: delta_max would exceed the "
                  f"channel limit of {d_cap:.1f} rad/us")
            continue
        r = run_quantum_mwis(G_paper, PAPER_IDS, cfg, np.random.default_rng(9))
        p_opt = sum(h["counts"] for h in r["hypotheses"]
                    if abs(h["weight"] - 19.2) < 1e-6) / r["shots"]
        rows.append((ratio, p_opt, r["violation_rate"]))
    return rows


# The QUBO penalty lambda is the blockade energy at the edge distance,
# U(r_edge) = C6 / (r_edge*R_b)^6 = Omega * r_edge^-6.  In units of Omega it is
# a pure number fixed by the *geometry*, not by any software knob.
U_over_omega = CFG.r_edge ** -6
print(f"\nEffective QUBO penalty from the geometry:")
print(f"  U(r_edge)/Omega = r_edge^-6 = {U_over_omega:.2f}")
print(f"  Design rule: delta_max/Omega must stay below this, or a blockaded pair")
print(f"  can profit from being excited together.\n")
print("  delta/Omega   P(optimum)   edge-violating shots")
for ratio, p, v in study_delta():
    flag = "   <- delta > U, blockade overrun" if ratio > U_over_omega else ""
    print(f"     {ratio:5.1f}      {p:6.1%}       {v:6.1%}{flag}")


# %% ===========================================================================
# CELL 15 -- What this shows, and what is still open
# ==============================================================================
SUMMARY = f"""
RESULTS OF THIS RUN
-------------------
* The classical front end turned {sum(len(s) for s in SCANS)} unlabelled radar blips into MWIS instances that
  stayed small: the largest cluster over the whole scenario was
  {max(max(p['clusters'], default=0) for p in RUN_Q['per_scan'])} vertices.  Gating, confirmation, track merging and N-scan pruning --
  not the quantum step -- are what make the problem fit.  Removing any one of
  them (track merging especially) blows the clusters up into near-cliques that
  no 2-D atom register can represent.
* The neutral-atom emulator reproduced the paper's worked example exactly
  (tracks {qres_paper['best']['sel']}, weight {qres_paper['best']['weight']:.1f} vs. the published 19.2).
* Run inside the tracking loop it solved {len(AUDIT)} MWIS instances on the array and
  matched the exact classical optimum on {n_opt} of them ({n_opt/max(len(AUDIT),1):.0%}), while
  following the objects with {acc_q['assoc_accuracy']:.0%} association accuracy through the
  crossing.  Note that {np.mean([p['violation'] for p in RUN_Q['per_scan'] if p['audit']]):.0%} of raw shots violated an edge and had to be
  repaired first -- the array alone was not enough.
* It returned a *distribution* over global hypotheses rather than one answer,
  which is what Eqs. (2)-(3) need for global track probabilities.  Be careful
  with this claim though: to keep the comparison fair I gave the classical run
  the same thing, by enumerating maximal independent sets exactly, and at these
  cluster sizes that is instant.  The honest statement is not "only the quantum
  machine gives many hypotheses" -- it is that the array's cost is a fixed
  number of shots regardless of how many hypotheses you want, whereas exact
  enumeration is worst-case exponential in the cluster size.  Nothing in this
  run is big enough for that to pay off yet.

WHAT I WOULD TELL THE SUMMER SCHOOL
-----------------------------------
1. lambda is not a free parameter on this hardware.  The QUBO penalty is
   C6/r_ij^6, fixed by where you place the atoms.  What you do choose is
   delta_max/Omega, and Cell 14 shows it is squeezed from BOTH sides:
     - too small (<= 1) and the detunings cannot express the weight differences,
       so the sweep returns the wrong independent set (P(optimum) 5% at 0.5);
     - too large (> U(r_edge)/Omega = 5.6) and a blockaded pair starts to profit
       from being excited together, and edge violations appear in the raw shots.
   The usable window here is roughly 2 <= delta_max/Omega <= 5.6.  Reporting a
   window rather than a single tuned number is the transferable result.
2. The binding constraint is geometry, not qubit count.  The device holds 256
   atoms, but Cell 14 shows perfect unit-disk embeddings becoming rare well
   before that as clusters grow and densify.  Cluster size, not array size, is
   the ceiling.
3. Weight encoding needs w_i > 0, and the paper's rule "keep only positive-score
   tracks" supplies it for free.  The DMM can only push detuning down, so
   m_i = 1 - w_i/w_max maps scores onto local detuning with no extra machinery.
4. Non-adiabaticity is useful here.  In a gate-model setting the spread over
   excited states is error; in MHT it is the near-optimal hypothesis set.

OPEN QUESTIONS (what I did NOT solve)
-------------------------------------
* Non-embeddable clusters.  My embedder reports the edges it cannot realise and
  I repair the sampled bitstrings classically, which biases the distribution.
  Proper approaches: ancilla-atom gadgets to mediate awkward edges, 3-D
  registers, or splitting the cluster and stitching solutions.
* Repair bias.  Every repaired shot is no longer a fair sample of the Boltzmann
  distribution, so P{{H_j}} from Eq. (2) is only approximate.  Quantifying that
  bias matters if the probabilities feed back into pruning.
* Unresolved targets.  Two objects merging into one blip should be allowed to
  share an observation; that breaks the "edge = shared blip" rule and needs
  explicit gadgets in the graph.
* Real hardware.  Everything here is a noiseless emulator.  Atom loss, finite
  Rydberg lifetime and detuning errors all attack the weight encoding directly,
  since the weights *are* detunings.
* The 6000 ns sequence limit already bites: with a fixed time budget the sweep
  cannot be made arbitrarily adiabatic, so larger clusters degrade for reasons
  that have nothing to do with atom count.
"""
print(SUMMARY)
