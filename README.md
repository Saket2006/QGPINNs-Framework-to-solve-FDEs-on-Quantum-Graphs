# QGPINNs: A Physics-Informed Neural Network Framework for Nonlocal Differential Equations on Quantum Graphs

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**QGPINNs** is a PyTorch based deep learning framework designed to solve fractional differential equations defined on quantum graphs. We primarily focus on elliptic and parabolic type differential equations, however the engine remains flexible to solve a wider variety of differential equations. The framework is modular and separates the QGPINNs engine, which handles the neural network architecture, from the run script, which defines the governing physics and topologies. A detailed paper on the methods and techniques used in the framework is available here.....

---

## Key Features

* **Modular Architecture**: Complete separation between core neural network engine (`Engine/`) and the user defined graph physics (`Section_3_Run_Scripts`, `Section_5_Run_Scripts`).
* **Fractional Caputo Operator**: The Caputo fractional operator is computed within the engine with $L1$ and $L2-1_{\sigma}$ schemes.
* **Graph Vertex Coupling**: Enforcement of Kirchhoff-Neumann transmission conditions and continuity constraints across junctions.
* **Singularity-Capturing Feature**: Learnable auxiliary feature $Z(t) = t^\xi$ to capture initial singularities.
* **Spectral Bias Mitigation**: Random Fourier Feature embeddings to resolve high-frequency spatial oscillations.
* **Dynamic Loss Balancing**: Integrated adaptive weighting mechanisms.
* **Inverse Problem Solver**: Built-in support for recovering fractional operator orders ($\alpha, \beta$) and physical parameters using noisy observation data.

---

## Repository Architecture

```text
QGPINNs/
├── Engine/
│   └── QGPINNs_Engine/      # Core neural network solver engines & fractional matrix assembly
├── Section_3_Run_Scripts/   # Theoretical benchmarks & ablation studies (Adaptive lambda, Fourier, Z(t))
├── Section_5_Run_Scripts/   # Real-world application scripts (Tadpole, Drainage Network, IEEE-14)
└── README.md
