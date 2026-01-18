# Flow Reconstruction

This project demonstrates flow-field reconstruction using diffusion-based methods. From left to right, the visualization shows the input field, the reconstructed output, and the ground-truth flow.

To assess generalization to out-of-distribution data, the model was evaluated on 256×256 tasks using only 1.5625% of the input information.

## Reference

Shu, D., Li, Z., & Farimani, A. B. A physics-informed diffusion model for high-fidelity flow field reconstruction. *arXiv:2211.14680v2*, 2023. https://arxiv.org/abs/2211.14680

Based on [NVIDIA PhysicsNemo Flow Reconstruction Diffusion](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/flow_reconstruction_diffusion)
