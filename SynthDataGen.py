#!/usr/bin/env python3
"""
===============================================================================
 Synthetic multi-object dataset for hard data association.
 Cell-Tracking-Challenge TIF layout, so any reader of the cell data reads this.
 2026 Niels Bohr Quantum Summer School -- SDU Odense
===============================================================================

ONE KNOB:  noise in [0, 1]

    noise = 0.0   clean    slow objects, no clutter, near-perfect detection
    noise = 0.5   moderate competing hypotheses appear
    noise = 1.0   severe   fast objects, heavy clutter, half the objects missed

It scales every imperfection together -- object speed (hence how far gates
overlap), clutter rate, missed-detection rate and pixel noise -- because in a
tracker these are not independent difficulties: they all end up as extra
candidate tracks competing for the same detections.

WHY SPEED IS PART OF "NOISE"
----------------------------
The number that decides whether data association is hard at all is

    ambiguity = (displacement per frame) / (nearest-neighbour spacing)

Below ~0.2 every gate holds exactly one candidate and there is no combinatorial
problem.  Real cell data sits at 0.03, which is why it turned out to need no
clever solver.  Raising `noise` raises this ratio.

WHAT TO WATCH IN THE DIAGNOSTIC
-------------------------------
Not the vertex count -- TREEWIDTH.  Exact classical dynamic programming solves an
MWIS instance in ~2^treewidth operations, so a 60-vertex cluster with treewidth 8
is solved exactly in 0.01 s.  Treewidth is the only difficulty measure that
matters, and it is what this generator tries to push up.

BUT WATCH THE `tw-clq` COLUMN -- AND THE RESULT IS NEGATIVE
-----------------------------------------------------------
A clique inflates treewidth for free, since treewidth >= maxclique - 1, without
making anything hard: a clique's MWIS is just its heaviest vertex.  It is also
the single worst structure for a 2-D unit-disk embedding.  So treewidth only
counts when it is NOT merely a clique, which is what `tw-clq` measures.

Running `sweep` shows the knob spanning ambiguity 0.16 -> 1.11 and treewidth
6 -> 23, but `tw-clq` stays <= 0 at every level.  The treewidth is ALWAYS
clique-driven.  That is not a defect of this generator; it is a property of MHT
conflict graphs -- every set of tracks competing for one detection is a clique,
so raising ambiguity grows cliques instead of creating hard structure.  Which
means the instances stay simultaneously easy to solve and hard to embed.

WHAT WAS TRIED, AND THE RESULT
------------------------------
Roughly 100 configurations across five distinct strategies were searched for an
instance with treewidth genuinely above the clique number:

  1. noise sweep, 0 -> 1                 tw-omega in {-1, 0}
  2. kinematic-exclusion edges           made it worse (proximity is
     (conflict = objects too close)      clique-forming; omega rose 10 -> 16)
  3. longer windows, fewer objects       best gap +3, exact MWIS 21 ms
  4. lattice formation (aiming for a     tw-omega = -1 in all 30 runs
     grid: omega ~ 2, tw ~ sqrt(n))
  5. density / gate / per-seed tuning    omega tracks MAX claimants exactly;
     to drive claimants-per-detection    low omega  -> small clusters, tw ~ 4
     down to ~2                          high omega -> tw = omega - 1

Best gap found anywhere: +3.  Slowest exact solve anywhere: 21 milliseconds.

THE MECHANISM
-------------
omega ~= the MAXIMUM number of tracks claiming any single detection.  Driving the
mean down does not help; omega follows the max.  And every conflict in this
formulation is mediated by two tracks passing through the same place at the same
time -- and co-location is a clique-forming relation.  Any conflict rule built on
shared spacetime produces cliques, hence near-chordality, hence tw = omega - 1.

CONCLUSION: physical tracking scenarios do not appear to generate hard MWIS
instances under the standard "conflict = shared observation" rule, at any
scenario setting.  The instances are simultaneously easy to solve classically
(tw = omega - 1) and hard to embed on a 2-D atom array (cliques above ~9 atoms
cannot be realised).  Both follow from the same clique structure.

So this generator is a good source of data that is hard to TRACK, and an honest
reporter of the fact that such data is not hard to OPTIMISE.  The two axes are
printed separately for exactly that reason.  Escaping the second one needs an
anti-transitive conflict rule (A-B and B-C conflict while A-C does not,
systematically), which spacetime co-location does not provide.

OUTPUT
------
    <out_dir>/<name>/<seq>/t000.tif ...            raw frames,  uint8
    <out_dir>/<name>/<seq>_GT/TRA/man_track000.tif labels,      uint16
    <out_dir>/<name>/<seq>_GT/TRA/man_track.txt    id start end parent
    <out_dir>/<name>.zip                           same tree, for zip readers

Use name="PhC-C2DL-PSC" to drop straight into scripts written for the cell data.

USAGE
-----
    python SyntheticMHTData.py            # noise = 0.6
    python SyntheticMHTData.py 0.9        # pick the noise level
    python SyntheticMHTData.py 0.6 sweep  # generate + compare several levels

    from SyntheticMHTData import make_dataset, diagnose
    make_dataset(noise=0.8)
===============================================================================
"""

from __future__ import annotations

import itertools
import os
import sys
import time
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    import networkx as nx
    from networkx.algorithms import approximation as _approx
except ImportError:                                    # diagnostic only
    nx = None

# ---- fixed scene constants (deliberately not knobs) -------------------------
HEIGHT, WIDTH = 576, 720
BG_LEVEL, BG_SHADING = 135, 12.0
BLOB_AMP, BLOB_LONG, BLOB_SHORT = 95.0, 3.4, 1.7
MARKER_RADIUS = 1
REGION_FRAC = 0.40            # objects confined to the middle of the frame:
                              # crowding is what makes gates overlap
TURN_SIGMA = 0.16             # rad/frame heading drift -> crossing paths


def _params(noise: float) -> Dict[str, float]:
    """Map the single knob onto every physical imperfection."""
    t = float(np.clip(noise, 0.0, 1.0))
    return {"speed": 2.5 + 14.0 * t,        # px/frame -> drives `ambiguity`
            "p_detect": 0.98 - 0.36 * t,    # missed detections
            "clutter": 36.0 * t,            # false alarms per frame
            "px_noise": 1.5 + 6.0 * t}      # sensor noise


# =============================================================================
#  Ground truth
# =============================================================================

def _motion(n_objects: int, n_frames: int, speed: float, rng
            ) -> Dict[int, Dict[int, np.ndarray]]:
    """Objects at constant speed with drifting heading, reflected in a box.

    Constant speed and slow turning keep trajectories crossing instead of
    dispersing, so many objects stay mutually gate-compatible for many frames --
    which is what makes the conflict graph interlock rather than fall apart into
    small independent families.
    """
    bw, bh = REGION_FRAC * WIDTH, REGION_FRAC * HEIGHT
    x0, y0 = (WIDTH - bw) / 2, (HEIGHT - bh) / 2
    pos = np.column_stack([rng.uniform(x0, x0 + bw, n_objects),
                           rng.uniform(y0, y0 + bh, n_objects)])
    ang = rng.uniform(0, 2 * np.pi, n_objects)
    spd = np.full(n_objects, speed)

    truth = {i + 1: {} for i in range(n_objects)}
    for k in range(n_frames):
        for i in range(n_objects):
            truth[i + 1][k] = pos[i].copy()
        ang += rng.normal(0, TURN_SIGMA, n_objects)
        spd = np.clip(spd + rng.normal(0, 0.08 * speed, n_objects),
                      0.35 * speed, 1.8 * speed)
        pos += np.column_stack([spd * np.cos(ang), spd * np.sin(ang)])
        for d, (lo, hi) in enumerate(((x0, x0 + bw), (y0, y0 + bh))):
            below, above = pos[:, d] < lo, pos[:, d] > hi
            pos[below, d] = 2 * lo - pos[below, d]
            pos[above, d] = 2 * hi - pos[above, d]
            flip = below | above
            ang[flip] = (np.pi - ang[flip]) if d == 0 else (-ang[flip])
    return truth


# =============================================================================
#  Rendering
# =============================================================================

def _blob(img: np.ndarray, x: float, y: float, ang: float, amp: float) -> None:
    r = int(np.ceil(3.2 * max(BLOB_LONG, BLOB_SHORT))) + 1
    xi0, xi1 = max(0, int(x) - r), min(WIDTH, int(x) + r + 1)
    yi0, yi1 = max(0, int(y) - r), min(HEIGHT, int(y) + r + 1)
    if xi0 >= xi1 or yi0 >= yi1:
        return
    yy, xx = np.mgrid[yi0:yi1, xi0:xi1]
    dx, dy = xx - x, yy - y
    ca, sa = np.cos(-ang), np.sin(-ang)
    u, v = dx * ca - dy * sa, dx * sa + dy * ca
    img[yi0:yi1, xi0:xi1] += amp * np.exp(
        -0.5 * ((u / BLOB_LONG) ** 2 + (v / BLOB_SHORT) ** 2))


def _frame(k: int, truth: Dict[int, Dict[int, np.ndarray]], p: Dict[str, float],
           rng) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    img = (BG_LEVEL + BG_SHADING * np.sin(2 * np.pi * xx / (2.3 * WIDTH))
           * np.cos(2 * np.pi * yy / (1.9 * HEIGHT))).astype(float)
    lab = np.zeros((HEIGHT, WIDTH), np.uint16)

    for oid, hist in truth.items():
        if k not in hist:
            continue
        x, y = hist[k]
        prev = hist.get(k - 1, hist[k])
        ang = np.arctan2(y - prev[1], x - prev[0]) if k - 1 in hist else 0.0

        # The GT marker is written even when the blob is not drawn: the object is
        # there, a missed detection is a sensor failure, not an absence.
        ry, rx = int(round(y)), int(round(x))
        if 0 <= ry < HEIGHT and 0 <= rx < WIDTH:
            m = MARKER_RADIUS
            lab[max(0, ry - m):ry + m + 1, max(0, rx - m):rx + m + 1] = oid
        if rng.random() <= p["p_detect"]:
            _blob(img, x, y, ang, BLOB_AMP * rng.uniform(0.85, 1.15))

    for _ in range(rng.poisson(p["clutter"])):
        _blob(img, rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT),
              rng.uniform(0, 2 * np.pi), BLOB_AMP * rng.uniform(0.6, 1.0))

    img += rng.normal(0, p["px_noise"], img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), lab


# =============================================================================
#  Generate
# =============================================================================

def make_dataset(noise: float = 0.6, out_dir: str = "SyntheticData",
                 name: str = "SYN-MHT", seq: str = "01", n_frames: int = 40,
                 n_objects: int = 55, seed: int = 0, make_zip: bool = True
                 ) -> Dict:
    """Write a synthetic dataset in Cell-Tracking-Challenge TIF layout."""
    p = _params(noise)
    rng = np.random.default_rng(seed)

    root = os.path.join(out_dir, name)
    raw_dir, tra_dir = os.path.join(root, seq), os.path.join(root, f"{seq}_GT", "TRA")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(tra_dir, exist_ok=True)

    truth = _motion(n_objects, n_frames, p["speed"], rng)
    for k in range(n_frames):
        img, lab = _frame(k, truth, p, rng)
        Image.fromarray(img).save(os.path.join(raw_dir, f"t{k:03d}.tif"))
        Image.fromarray(lab).save(os.path.join(tra_dir, f"man_track{k:03d}.tif"))

    with open(os.path.join(tra_dir, "man_track.txt"), "w") as f:
        f.write("".join(f"{oid} {min(h)} {max(h)} 0\n"
                        for oid, h in sorted(truth.items()) if h))

    zip_path = None
    if make_zip:
        zip_path = os.path.join(out_dir, f"{name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for folder, _, files in os.walk(root):
                for fn in sorted(files):
                    full = os.path.join(folder, fn)
                    z.write(full, os.path.relpath(full, out_dir))

    return {"noise": noise, "params": p, "root": root, "zip": zip_path,
            "out_dir": out_dir, "name": name, "seq": seq,
            "n_frames": n_frames, "n_objects": n_objects}


# =============================================================================
#  Read back
# =============================================================================

def load_frame(info: Dict, k: int) -> np.ndarray:
    return np.array(Image.open(os.path.join(
        info["out_dir"], info["name"], info["seq"], f"t{k:03d}.tif")))


def load_gt(info: Dict, k: int) -> List[Tuple[int, np.ndarray]]:
    lab = np.array(Image.open(os.path.join(
        info["out_dir"], info["name"], f"{info['seq']}_GT", "TRA",
        f"man_track{k:03d}.tif")))
    ids = np.unique(lab)
    ids = ids[ids > 0]
    cen = ndi.center_of_mass(np.ones_like(lab), lab, ids)
    return [(int(i), np.array(p[::-1])) for i, p in zip(ids, cen)]


# =============================================================================
#  Diagnostic
# =============================================================================

def _detect(img: np.ndarray) -> np.ndarray:
    """Identical detector to the one used on the real cell data."""
    f = img.astype(float)
    d = f - ndi.gaussian_filter(f, 15.0)
    mask = ndi.binary_opening(d > 2.5 * d[d > 0].std(), np.ones((2, 2)))
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros((0, 2))
    areas = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if 6 <= areas[i] <= 400]
    if not keep:
        return np.zeros((0, 2))
    return np.array([q[::-1] for q in
                     ndi.center_of_mass(np.ones_like(lab), lab, keep)])


def _conflict_graph(dets: List[np.ndarray], per_seed: int = 3, min_hits: int = 3,
                    exclusion: float = 0.0):
    """Compact MHT front end: gated track hypotheses -> conflict graph.

    `exclusion` > 0 adds KINEMATIC-EXCLUSION edges on top of the usual
    shared-detection edges: two tracks also conflict if they would place two
    distinct physical objects closer than `exclusion` pixels at the same instant
    (objects cannot pass through one another).

    The idea was structural.  Shared-detection edges make the graph a union of
    cliques -- one per detection -- which is (near-)chordal, and chordal graphs
    have treewidth exactly omega-1, i.e. no difficulty beyond their largest
    clique.  An exclusion edge belongs to NO detection-clique, so it should be
    able to close a chordless cycle and lift treewidth above the clique number.

    MEASURED RESULT: it does not work, and it makes things slightly worse.
    Proximity is a transitive-ish relation -- if A is near B and B near C then A
    is usually near C -- so exclusion edges arrive in clumps, i.e. as MORE
    cliques.  Raising `exclusion` therefore raises omega and drives tw - omega
    back to -1:

        excl:      0     6    10    14    20        (noise 0.35, 55 objects)
        omega:    10    10    10    14    16
        tw-omega: +3    +3    +3    -1    -1

    The largest gap found anywhere was +3, at exclusion = 0, in a large SPARSE
    cluster (54 vertices, density 0.17) -- i.e. the baseline already produces the
    least-chordal instances, and adding geometric edges only densifies them.
    Exact MWIS never took more than 21 ms in any configuration.

    Kept here as evidence, and because the parameter is the right place to try
    other conflict definitions.  What would be needed is an ANTI-transitive
    relation (A-B and B-C conflict while A-C does not, systematically), which is
    what builds long chordless cycles.  Geometric tracking data does not seem to
    produce one naturally.
    """
    beta = 0.18 * np.mean([len(d) for d in dets]) / (WIDTH * HEIGHT)
    sig_d, sig_q, pd, gate = 2.5, 7.0, 0.8, 9.21
    live = [(((0, j),), 0.0, q, 0) for j, q in enumerate(dets[0])]
    for fi in range(1, len(dets)):
        nxt = []
        for hist, sc, pos, last in live:
            var = sig_d ** 2 + sig_q ** 2 * (fi - last)
            for j, q in enumerate(dets[fi]):
                d2 = float(np.sum((q - pos) ** 2) / var)
                if d2 > gate:
                    continue
                nxt.append((hist + ((fi, j),),
                            sc + np.log(pd) - np.log(beta) - np.log(2 * np.pi)
                            - np.log(2 * var) - 0.5 * d2, q, fi))
            nxt.append((hist + ((fi, -1),), sc + np.log(1 - pd), pos, last))
        by_seed: Dict[int, List] = {}
        for t in nxt:
            by_seed.setdefault(t[0][0][1], []).append(t)
        live = []
        for ts in by_seed.values():
            ts.sort(key=lambda t: -t[1])
            live.extend(ts[:per_seed])

    best: Dict[frozenset, Tuple] = {}
    for hist, sc, _, _ in live:
        key = frozenset((f, j) for f, j in hist if j >= 0)
        if len(key) < min_hits or sc <= 0:
            continue
        if key not in best or sc > best[key][1]:
            best[key] = (hist, sc)
    tracks = list(best.values())

    G = nx.Graph()
    for i, (_, sc) in enumerate(tracks):
        G.add_node(i, weight=float(sc))
    by_det: Dict[Tuple[int, int], List[int]] = {}
    for i, (hist, _) in enumerate(tracks):
        for (f, j) in hist:
            if j >= 0:
                by_det.setdefault((f, j), []).append(i)
    for members in by_det.values():
        for a, b in itertools.combinations(members, 2):
            G.add_edge(a, b)

    G.graph["claimants"] = [len(m) for m in by_det.values() if len(m) >= 2]

    if exclusion > 0:
        pos = [{f: dets[f][j] for f, j in hist if j >= 0} for hist, _ in tracks]
        for i, j in itertools.combinations(range(len(tracks)), 2):
            if G.has_edge(i, j):
                continue
            for f in set(pos[i]) & set(pos[j]):
                if np.linalg.norm(pos[i][f] - pos[j][f]) < exclusion:
                    G.add_edge(i, j)
                    break
    return G


def diagnose(info: Dict, window: Sequence[int] = (0, 1, 2, 3, 4, 5),
             verbose: bool = True, exclusion: float = 0.0) -> Dict:
    """Is this dataset actually hard?  Reports ambiguity and treewidth."""
    dets = [_detect(load_frame(info, k)) for k in window]
    gts = [load_gt(info, k) for k in window]

    recs, precs = [], []
    for D, G in zip(dets, gts):
        if len(D) == 0 or len(G) == 0:
            continue
        used, tp = set(), 0
        for _, g in G:
            d = np.linalg.norm(D - g, axis=1)
            j = int(np.argmin(d))
            if d[j] <= 12.0 and j not in used:
                used.add(j)
                tp += 1
        recs.append(tp / len(G))
        precs.append(tp / len(D))

    a, b = dict(gts[0]), dict(gts[1])
    common = set(a) & set(b)
    disp = np.array([np.linalg.norm(b[i] - a[i]) for i in common])
    P = np.array([q for _, q in gts[0]])
    nn = np.array([np.min(np.delete(np.linalg.norm(P - P[i], axis=1), i))
                   for i in range(len(P))])
    amb = float(np.median(disp) / np.median(nn))

    out = {"noise": info["noise"], "ambiguity": amb,
           "recall": float(np.mean(recs)), "precision": float(np.mean(precs)),
           "median_disp": float(np.median(disp)), "median_nn": float(np.median(nn)),
           "dets_per_frame": [len(d) for d in dets], "treewidth": None}

    if nx is not None:
        G = _conflict_graph(dets, exclusion=exclusion)
        clusters = [sorted(c) for c in nx.connected_components(G)]
        rows = []
        for c in sorted(clusters, key=len, reverse=True)[:6]:
            S = G.subgraph(c)
            n = len(c)
            tw, _ = _approx.treewidth_min_degree(S)
            t0 = time.time()
            comp = nx.complement(S)
            for v in comp.nodes:
                comp.nodes[v]["weight"] = int(round(G.nodes[v]["weight"] * 1000))
            nx.max_weight_clique(comp, weight="weight")
            rows.append({"n": n,
                         "density": (2 * S.number_of_edges() / (n * (n - 1))) if n > 1 else 0,
                         "max_clique": max(len(q) for q in nx.find_cliques(S)),
                         "treewidth": tw, "exact_s": time.time() - t0})
        out.update({"n_vertices": G.number_of_nodes(), "n_clusters": len(clusters),
                    "top_clusters": rows,
                    "treewidth": max(r["treewidth"] for r in rows) if rows else 0})

    if verbose:
        verdict = ("trivial" if amb < 0.20 else
                   "usable" if amb < 0.45 else
                   "hard" if amb < 0.80 else "chaotic")
        print(f"\n--- diagnostic: noise = {info['noise']:.2f} ---")
        print(f"  detections/frame  : {out['dets_per_frame']}")
        print(f"  detector recall   : {out['recall']:.2f}   "
              f"precision {out['precision']:.2f}")
        print(f"  displacement/gap  : {out['median_disp']:.1f} / "
              f"{out['median_nn']:.1f} px")
        print(f"  AMBIGUITY         : {amb:.2f}  ({verdict})")
        if out["treewidth"] is not None:
            print(f"  conflict graph    : {out['n_vertices']} vertices, "
                  f"{out['n_clusters']} clusters")
            print(f"    {'n':>4} {'density':>8} {'maxclq':>7} {'treewidth':>10} "
                  f"{'exact MWIS':>12}")
            for r in out["top_clusters"]:
                print(f"    {r['n']:>4} {r['density']:>8.2f} {r['max_clique']:>7} "
                      f"{r['treewidth']:>10} {r['exact_s']:>11.3f}s")
            tw = out["treewidth"]
            mc = max(r["max_clique"] for r in out["top_clusters"])
            cl = G.graph.get("claimants", [])
            if cl:
                print(f"  claimants/detection: mean {np.mean(cl):.1f}, "
                      f"max {max(cl)}  <- this sets the clique number")
                out["max_claimants"] = int(max(cl))
            # A clique inflates treewidth for free (tw >= maxclique - 1) without
            # making the instance hard: a clique's MWIS is just its heaviest
            # vertex.  It is also the worst possible case for a unit-disk
            # embedding.  So high treewidth only counts when it is NOT simply a
            # clique -- that is the number to push up.
            clique_driven = tw <= mc
            print(f"  treewidth {tw}, largest clique {mc}"
                  f"  -> exact DP ~2^{tw} = {2.0**tw:.1e} ops")
            slowest = max(r["exact_s"] for r in out["top_clusters"])
            print()
            print(f"  VERDICT (two independent axes)")
            print(f"    tracking difficulty  : {verdict.upper():8s} "
                  f"(ambiguity {amb:.2f})")
            if clique_driven:
                print(f"    optimisation difficulty: TRIVIAL  (treewidth is just "
                      f"the clique; exact MWIS {slowest*1000:.0f} ms)")
            elif tw < 20:
                print(f"    optimisation difficulty: TRIVIAL  (exact MWIS "
                      f"{slowest*1000:.0f} ms)")
            else:
                print(f"    optimisation difficulty: NON-TRIVIAL -- treewidth not "
                      f"explained by a clique")
            print("  These two are independent: data can be very hard to TRACK")
            print("  while the resulting MWIS instances stay trivial to SOLVE.")
            out["clique_driven"] = bool(clique_driven)
            out["max_clique"] = int(mc)
    return out


# =============================================================================
#  Benchmark export -- instances a PC-emulated quantum solver can actually run
# =============================================================================

def export_benchmark(path: str = "mwis_benchmark.json", noise: float = 0.45,
                     n_objects: int = 400, n_frames: int = 12, window: int = 8,
                     gate_chi2: float = 3.0, sig_q: float = 4.0, per_seed: int = 2,
                     max_atoms: int = 20, max_clique: int = 8, seed: int = 0,
                     verbose: bool = True) -> List[Dict]:
    """Emit MWIS instances that a neutral-atom solver on a PC can really run.

    Two hard limits decide what is usable, and they pull in opposite directions:

      n <= max_atoms   because a state-vector emulator costs 2^n
      omega <= ~9      because that many atoms cannot all sit pairwise inside one
                       blockade radius while staying >= 5 um apart, so a bigger
                       clique simply cannot be embedded in 2-D

    The tight-gate settings below keep omega small (few tracks contest any one
    detection), which is what makes the instances embeddable.  Note these
    instances are NOT computationally hard -- exact MWIS solves them in
    milliseconds -- so they are a correctness-and-quality benchmark, not evidence
    of advantage.  Each instance ships with its exact optimum for scoring.
    """
    import json
    info = make_dataset(noise=noise, n_objects=n_objects, n_frames=n_frames,
                        name="BENCH", seed=seed, make_zip=False)
    dets = [_detect(load_frame(info, k)) for k in range(window)]
    G = _conflict_graph_gated(dets, per_seed, 3, gate_chi2, sig_q)

    out = []
    for c in sorted((sorted(c) for c in nx.connected_components(G)),
                    key=len, reverse=True):
        if not (6 <= len(c) <= max_atoms):
            continue
        sub = G.subgraph(c)
        w = max(len(q) for q in nx.find_cliques(sub))
        if w > max_clique:
            continue
        idx = {v: i for i, v in enumerate(c)}
        comp = nx.complement(sub)
        for v in comp.nodes:
            comp.nodes[v]["weight"] = int(round(G.nodes[v]["weight"] * 1000))
        clique, _ = nx.max_weight_clique(comp, weight="weight")
        tw, _ = _approx.treewidth_min_degree(sub)
        out.append({
            "n": len(c),
            "weights": [round(G.nodes[v]["weight"], 6) for v in c],
            "edges": sorted([idx[a], idx[b]] for a, b in sub.edges),
            "max_clique": int(w),
            "treewidth": int(tw),
            "optimum_weight": round(sum(G.nodes[v]["weight"] for v in clique), 6),
            "optimum_set": sorted(idx[v] for v in clique)})

    with open(path, "w") as f:
        json.dump({"source": "SyntheticMHTData.export_benchmark",
                   "noise": noise, "n_objects": n_objects, "window": window,
                   "instances": out}, f, indent=1)
    if verbose:
        print(f"\nwrote {len(out)} benchmark instances -> {path}")
        print(f"  {'n':>4} {'edges':>6} {'omega':>6} {'tw':>4} {'|MWIS|':>7} "
              f"{'optimum':>9}")
        for r in out:
            print(f"  {r['n']:>4} {len(r['edges']):>6} {r['max_clique']:>6} "
                  f"{r['treewidth']:>4} {len(r['optimum_set']):>7} "
                  f"{r['optimum_weight']:>9.2f}")
        print("  every instance: n <= %d atoms, clique <= %d (2-D embeddable),"
              % (max_atoms, max_clique))
        print("  exact optimum included for scoring.")
    return out


def _conflict_graph_gated(dets, per_seed, min_hits, gate_chi2, sig_q):
    """_conflict_graph with the gate exposed (tight gates keep omega small)."""
    beta = 0.18 * np.mean([len(d) for d in dets]) / (WIDTH * HEIGHT)
    sig_d, pd = 2.5, 0.8
    live = [(((0, j),), 0.0, q, 0) for j, q in enumerate(dets[0])]
    for fi in range(1, len(dets)):
        nxt = []
        for hist, sc, pos, last in live:
            var = sig_d ** 2 + sig_q ** 2 * (fi - last)
            for j, q in enumerate(dets[fi]):
                d2 = float(np.sum((q - pos) ** 2) / var)
                if d2 > gate_chi2:
                    continue
                nxt.append((hist + ((fi, j),),
                            sc + np.log(pd) - np.log(beta) - np.log(2 * np.pi)
                            - np.log(2 * var) - 0.5 * d2, q, fi))
            nxt.append((hist + ((fi, -1),), sc + np.log(1 - pd), pos, last))
        by_seed: Dict[int, List] = {}
        for t in nxt:
            by_seed.setdefault(t[0][0][1], []).append(t)
        live = []
        for ts in by_seed.values():
            ts.sort(key=lambda t: -t[1])
            live.extend(ts[:per_seed])
    best: Dict[frozenset, Tuple] = {}
    for hist, sc, _, _ in live:
        key = frozenset((f, j) for f, j in hist if j >= 0)
        if len(key) < min_hits or sc <= 0:
            continue
        if key not in best or sc > best[key][1]:
            best[key] = (hist, sc)
    tracks = list(best.values())
    G = nx.Graph()
    for i, (_, sc) in enumerate(tracks):
        G.add_node(i, weight=float(sc))
    by_det: Dict[Tuple[int, int], List[int]] = {}
    for i, (hist, _) in enumerate(tracks):
        for (f, j) in hist:
            if j >= 0:
                by_det.setdefault((f, j), []).append(i)
    for m in by_det.values():
        for a, b in itertools.combinations(m, 2):
            G.add_edge(a, b)
    return G


# =============================================================================
if __name__ == "__main__":
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    if len(sys.argv) > 2 and sys.argv[2] == "bench":
        export_benchmark()
    elif len(sys.argv) > 2 and sys.argv[2] == "sweep":
        print(f"{'noise':>6} {'ambig':>7} {'recall':>7} {'verts':>6} "
              f"{'tw':>4} {'maxclq':>7} {'tw-clq':>7} {'exactMWIS':>10}")
        print("  (tw-clq > 0 means the treewidth is NOT just a clique: "
              "that is the useful regime)")
        for nz in (0.0, 0.25, 0.5, 0.75, 1.0):
            inf = make_dataset(noise=nz, name=f"SYN-MHT-{int(nz*100):03d}",
                               make_zip=False)
            d = diagnose(inf, verbose=False)
            rows = d.get("top_clusters", [])
            slow = max((r["exact_s"] for r in rows), default=0.0)
            # the best cluster is the one with most treewidth ABOVE its clique
            gap = max((r["treewidth"] - r["max_clique"] for r in rows), default=0)
            mc = max((r["max_clique"] for r in rows), default=0)
            print(f"{nz:6.2f} {d['ambiguity']:7.2f} {d['recall']:7.2f} "
                  f"{d.get('n_vertices', 0):6d} {d['treewidth']:4} {mc:7} "
                  f"{gap:+7} {slow:9.3f}s")
    else:
        print(f"generating with noise = {noise} ...")
        info = make_dataset(noise=noise)
        print(f"  {info['root']}")
        print(f"  {info['zip']}")
        diagnose(info)
