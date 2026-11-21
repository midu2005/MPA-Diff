# MPA-Diff

**MPA-Diff: An Intelligent Forecasting System with Historical Knowledge Awareness and Adaptive Diffusion**

MPA-Diff is a probabilistic time series forecasting framework that integrates multi-period historical representations, pathwise autoregressive state propagation, and prior-centered adaptive diffusion.

## Architecture

The framework consists of three main components:

- **Multi-Period Representation (MP):** Extracts historical information across multiple temporal windows and constructs a deterministic forecasting prior.
- **Pathwise Autoregressive Propagation (AR):** Updates trajectory-specific hidden states and dynamically retrieves relevant historical representations.
- **Prior-Centered Adaptive Diffusion (PCAD):** Adapts diffusion scales according to trajectory dynamics and generates future samples conditioned on the deterministic prior.

Generated samples are recursively incorporated into their respective trajectories to support multi-step probabilistic forecasting.

## Code

`mpa_diff.py` provides the core architecture and computational components of MPA-Diff.

## Requirements

- Python
- PyTorch