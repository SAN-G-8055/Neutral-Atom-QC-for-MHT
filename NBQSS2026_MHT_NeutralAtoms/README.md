# Data Association for MHT on a Neutral-Atom Quantum Computer

2026 Niels Bohr Quantum Summer School, SDU Odense — Challenge 7.

Self-contained package: code, figures, a short paper, and an A1 conference poster.

## Contents

```
code/
  NielsBohrProject.py            full pipeline, 15 `#%%` cells (~2 min)
  Route1_HypothesisDiscovery.py  hypothesis-discovery experiment (~15 min)
  requirements.txt               pinned package versions
figures/                         all figures, as .pdf (for LaTeX) and .png
report/report.tex / report.pdf   4-page paper, A4
poster/poster.tex  / poster.pdf  one-page conference poster, A1 portrait
```

## Before you present: edit the author blocks

Both documents ship with four placeholder authors.

- **Poster** — `poster/poster.tex`, the block marked
  `HEADER --- EDIT AUTHORS AND AFFILIATIONS HERE` (near the top of the document
  body). Replace `Author One`…`Author Four` and the four affiliation lines.
- **Report** — `report/report.tex`, the `\author{...}` command.

## Rebuilding

Documents (needs `pdflatex`; figure paths are relative, so build in place):

```bash
cd report && pdflatex report.tex && pdflatex report.tex
cd poster && pdflatex poster.tex
```

Code (needs the venv in the parent project folder):

```bash
../.venv/bin/python code/NielsBohrProject.py
../.venv/bin/python code/Route1_HypothesisDiscovery.py
```

`Route1_HypothesisDiscovery.py` reads `Train Data.zip` directly from the parent
folder — nothing is extracted, but the zip must be present and the script must be
run with the parent folder as the working directory.

Both scripts are deterministic: the emulator's shot noise is seeded.

## What the results are

- The physics mapping works. The array reproduced the published benchmark of
  Papageorgiou & Salpukas exactly (tracks {3,6}, weight 19.2) and matched the
  exact classical optimum on **21 of 21** MWIS instances inside the tracking loop.
- The proposed sampling advantage **did not materialise**. On real microscopy
  data the array needed 27.5 samples to capture 90% of the posterior against
  23.6 for simulated annealing — no gain, though the margin is small and it won
  on 4 of 6 instances individually.
- The bottleneck is **embeddability**, not qubit count: real MHT conflict graphs
  are not unit-disk graphs, so 34% of measurements violated a constraint and had
  to be repaired classically. We never needed more than 12 of 256 atoms.

Data: `PhC-C2DL-PSC`, ISBI Cell Tracking Challenge (celltrackingchallenge.net).
Quantum emulation: Pulser + QuTiP, `WeightedAnalogDevice`.
