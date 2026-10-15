# blochsphere

Interactive dashboard for watching a quantum neural network encode classical data, move a qubit state across the Bloch sphere, and train.

The first module is a **dynamic Bloch sphere trajectory tracker**: load a data point, pass it through parameterized rotations \(R_x(\theta)\) and \(R_y(\theta)\), and draw the path of the state vector on the sphere. Later modules add the loss landscape and the gradient-descent loop.

## Reference

The design follows [VIOLET: Visual Analytics for Explainable Quantum Neural Networks](https://arxiv.org/abs/2312.15276) (Ruan et al., arXiv:2312.15276). VIOLET splits a QNN into three views:

- **Encoder View** — classical inputs become quantum states (this repo's feature map and Bloch trajectory).
- **Ansatz View** — how variational parameters move the state while training.
- **Feature View** — what the trained circuit has learned, read out from measurements.

blochsphere starts with the encoder trajectory on the Bloch sphere, then connects that path to the ansatz and the loss surface.

## Status

Repository created. Design, implementation plan, and the Bloch sphere tracker are in progress.
