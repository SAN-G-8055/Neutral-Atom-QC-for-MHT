# Neutral-Atom Quantum Computing for Multiple Hypothesis Tracking

This repository contains code developed for our **Niels Bohr Quantum Summer School 2026** project on solving **Multiple Hypothesis Tracking (MHT)** data-association problems using neutral-atom quantum computing.

## Problem

Multiple Hypothesis Tracking is used to reconstruct target trajectories from a sequence of noisy and unlabelled measurements. Instead of committing immediately to a single measurement-to-target association, MHT maintains several competing candidate tracks and later determines which combination provides the most plausible global explanation of the observations.

The global data-association problem can be formulated as a **Maximum Weight Independent Set (MWIS)** problem. Each candidate track is represented by a weighted vertex in a conflict graph, where the weight reflects the track likelihood. An edge connects two candidate tracks that are mutually incompatible, for example because they share a detection or belong to the same initiating track family.

The objective is therefore to select the highest-weight collection of mutually compatible candidate tracks,

$$
\max_{\mathbf{x}} \sum_i w_i x_i
$$

subject to

$$
x_i+x_j \leq 1,
\qquad
\forall (i,j)\in E.
$$

In this project, the MWIS problem is mapped to a programmable **neutral-atom quantum system**. Candidate tracks are represented by atoms, track weights are encoded through site-dependent detuning, and incompatibilities are represented through distance-dependent Rydberg interactions.

The neutral-atom optimization is implemented and simulated using **Pasqal's Pulser** framework. We investigate the approach using both controlled synthetic tracking scenarios and real automotive radar data from the **RADIATE** dataset.

The project also explores an important limitation of this mapping: an arbitrary conflict graph cannot necessarily be represented exactly by the physical \(C_6/R^6\) interaction geometry of a two-dimensional neutral-atom array.

---

## Authors

- **Sanidhya Gupta** — Durham University
- **Natan Karaev** — Technion — Israel Institute of Technology
- **Marissa McMaster** — University of Maryland, College Park
- **Matt Prest** — City University of New York

Developed as part of the **Niels Bohr Quantum Summer School 2026**.

---

## References

### Multiple Hypothesis Tracking and MWIS

D. J. Papageorgiou and M. R. Salpukas,  
**"The Maximum Weight Independent Set Problem for Data Association in Multiple Hypothesis Tracking,"**  
in *Optimization and Cooperative Control Strategies*, Springer, 2009.  
https://doi.org/10.1007/978-3-540-88063-9_15

### Pulser

H. Silvério et al.,  
**"Pulser: An Open-Source Package for the Design of Pulse Sequences in Programmable Neutral-Atom Arrays,"**  
*Quantum*, vol. 6, p. 629, 2022.  
https://doi.org/10.22331/q-2022-01-24-629

Pulser documentation:  
https://docs.pasqal.com/pulser/

Pulser MWIS tutorial:  
https://docs.pasqal.com/pulser/tutorials/mwis/

### RADIATE Dataset

M. Sheeny et al.,  
**"RADIATE: A Radar Dataset for Automotive Perception in Bad Weather,"**  
2021 IEEE International Conference on Robotics and Automation (ICRA), 2021.  
https://doi.org/10.1109/ICRA48506.2021.9562089

RADIATE dataset:  
http://pro.hw.ac.uk/radiate/
