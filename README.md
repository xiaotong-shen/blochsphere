# blochsphere

Interactive dashboard for a small quantum neural network. Load a classical number, watch angle encoding and two variational rotations carry the state across the Bloch sphere, then step gradient descent on the loss surface.

A second view prepares two qubits and shows entanglement shorten both reduced Bloch vectors, pulling them inside the ball.

## Run

```bash
npm install
npm test
npm run dev
```

The dev server prints a local URL. `npm run build` typechecks and writes `dist/`.

## Circuit

```
|0⟩ —— Ry(x) —— Ry(θ) —— Rx(φ) —— ⟨Z⟩
```

`x` is the inspected input. Training minimises the mean squared error of `⟨Z⟩` against six fixed labels. The sphere follows the inspected input. The loss map uses the whole batch. Gradients use the parameter-shift rule.

The entanglement circuit is `|00⟩`, local `Ry` on each qubit, then `CRy(γ)` controlled by qubit 0. For a pure two-qubit state the reduced radius is `√(1 − C²)`.

## Docs

- [Design](docs/design.md) — conventions, the VIOLET split, and the decisions this base is built on
- [Implementation plan](docs/implementation-plan.md) — what is already in the app, and the later views

## Reference

The information architecture follows [VIOLET: Visual Analytics for Explainable Quantum Neural Networks](https://arxiv.org/abs/2312.15276) (Ruan et al.). VIOLET separates encoding, the ansatz, and measurement. This repository draws that story on the Bloch ball. It does not reproduce VIOLET's satellite chart or augmented heatmap. Those are later phases in the plan.

Statevector evolution for one and two qubits runs in the browser. There is no PennyLane or TorchQuantum dependency in this base.
