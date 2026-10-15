# Design: QNN state-space visualiser

blochsphere is an interactive simulator for one question: how does a small quantum neural network move a state while it encodes data and trains?

The geometric object is the Bloch ball. A pure single-qubit state is a point on the unit sphere. A mixed single-qubit state, including the reduced state of an entangled qubit, is a point inside the ball. The length of that vector is what the display tracks.

This is not a port of VIOLET. [VIOLET](https://arxiv.org/abs/2312.15276) (Ruan et al., arXiv:2312.15276) is the reference for what a QNN explanation has to show. Its own pictures are a satellite chart (basis states and rotation angles) and an augmented heatmap (measurements). blochsphere keeps VIOLET's three-layer split and answers the geometric half of that problem with the Bloch ball.

## What VIOLET contributes

VIOLET divides a QNN into data encoding, an ansatz, and measurement (their Figure 1). From expert interviews they keep six requirements:

| Id | Requirement | Where it lives here |
| --- | --- | --- |
| R1 | Show how classical data becomes a quantum state | Encoder arc on the sphere, plus basis probabilities |
| R2 | Show what a rotation angle does, including across training steps | Live θ, φ and the gradient step |
| R3 | Follow one data point through the circuit | Gate-coloured trajectory and the circuit strip |
| R4 | Make the measurement readable and check normalisation | ⟨Z⟩, P(0), P(1), and P(0)+P(1) = 1 |
| R5 | Show the feature the model has learned | Per-point ⟨Z⟩ swatches. The augmented heatmap is later |
| R6 | Show loss, accuracy, and parameter history | Loss number, sparkline, and the path on the (θ, φ) map |

Their source demo is at https://violet-source.github.io/. Case studies there use TorchQuantum. This project does not depend on that stack for circuits that fit in an exact statevector.

## Circuit in the base

One qubit, angle encoding, then a two-parameter hardware-efficient ansatz. Angles are radians.

```
|0⟩ —— Ry(x) —— Ry(θ) —— Rx(φ) —— measure Z
```

- `x` is the inspected classical input.
- `Ry(x)` is the feature map. Angle encoding is the method VIOLET treats as the usual encoder.
- `Ry(θ)` and `Rx(φ)` are the trainable ansatz.
- The prediction is the expectation ⟨Z⟩ ∈ [−1, 1].

Training points are six fixed inputs, three labelled +1 and three labelled −1:

```
x ∈ {−1.2, −0.75, −0.35} → y = +1
x ∈ {+0.35, +0.75, +1.2} → y = −1
```

Loss is the mean squared error between ⟨Z⟩ and the label. The sphere always shows the inspected `x`. The loss and the gradient always use the six training points. Those are different jobs and the interface says so.

A second view prepares a pure two-qubit state

```
|00⟩ —— Ry(θ₀) on q₀ —— Ry(θ₁) on q₁ —— CRy(γ) —— reduced states
```

`CRy(γ)` is controlled by qubit 0 and acts on qubit 1. At `θ₀ = π/2`, `θ₁ = 0`, `γ = π` the state is the Bell state `(|00⟩ + |11⟩) / √2` up to the standard Ry action. Each reduced Bloch vector then sits at the origin.

## State and gate conventions

Global phase is ignored by the Bloch vector. The computational state is kept anyway so the readout can show amplitudes.

A single-qubit state `α|0⟩ + β|1⟩` has

```
r_x = 2 Re(α* β)
r_y = 2 Im(α* β)
r_z = |α|² − |β|²
```

with `α*` the complex conjugate of `α`. North pole `r_z = +1` is `|0⟩`.

Gates match the usual textbook generators, the same convention as Qiskit:

```
Rx(α) = [[cos(α/2), −i sin(α/2)], [−i sin(α/2), cos(α/2)]]
Ry(α) = [[cos(α/2), −sin(α/2)], [sin(α/2), cos(α/2)]]
Rz(α) = [[e^{−iα/2}, 0], [0, e^{iα/2}]]
```

On the Bloch ball these are right-handed rotations by `α` about the named axis. `Ry(π/2)|0⟩ = |+⟩` and `Rx(π/2)|0⟩ = |−i⟩`.

Two-qubit amplitudes use qubit 0 as the least significant bit: index `q0 + 2 q1`, so the basis order is `|00⟩, |01⟩, |10⟩, |11⟩`. The reduced state of one qubit is the partial trace. Its Bloch coordinates come from `ρ = (I + r · σ) / 2`:

```
r_x = 2 Re(ρ₀₁)
r_y = −2 Im(ρ₀₁)
r_z = ρ₀₀ − ρ₁₁
```

Purity is `Tr(ρ²)`. Concurrence of a pure two-qubit state is `C = 2 |c₀₀ c₁₁ − c₀₁ c₁₀|`. For any pure two-qubit state the reduced radius satisfies `|r| = √(1 − C²)`. Entanglement does not move the point along the sphere. It shortens the arrow.

## Scene coordinates

Three.js is Y-up. The Bloch ball is Z-up. The map

```
(r_x, r_y, r_z) → (r_x, r_z, −r_y)
```

keeps a right-handed frame: Bloch Z is up, Bloch X is to the right, and Bloch Y points into the screen in the default camera. A unit test checks that the images of the Bloch basis still have `x̂ × ŷ = ẑ`.

Trajectories are not splines drawn between endpoints. Each gate is sampled by applying `R(tα)` for `t` from 0 to 1 to the state at the start of that gate. The painted curve is the image of those states. Playback reveals the three stages in order: encode, `Ry(θ)`, `Rx(φ)`.

## Training and the loss map

The parameter-shift rule for `Rx` and `Ry` is exact:

```
∂⟨Z⟩/∂α = (⟨Z⟩(α + π/2) − ⟨Z⟩(α − π/2)) / 2
```

The loss derivative follows by the chain rule through `L = mean (⟨Z⟩ − y)²`. One gradient step is

```
(θ, φ) ← (θ, φ) − η ∇L
```

The map under the sphere is `L(θ, φ)` on `[−π, π]²` for the fixed training set. It does not depend on the inspected `x`. The marker is the current parameter. The polyline is the sequence of gradient steps. Dragging a slider moves the marker and does not append a step.

This base has two parameters, so the landscape is a heatmap rather than a 3D surface. A third parameter would be shown by slicing it, not by replacing this map.

## Interface

One dark instrument layout:

- Left: mode, inspected input, ansatz angles, playback, and training.
- Centre: the Bloch ball. QNN mode draws one pure-state arrow. Entangle mode draws two reduced arrows, which may end inside the ball.
- Right: amplitudes, Bloch coordinates, probabilities, loss, and a sparkline.
- Bottom: the `(θ, φ)` loss map in QNN mode, or concurrence and the radius identity in Entangle mode.

Colour is reserved for meaning. Blue is the encoder arc and the Z axis. Amber is `Ry(θ)`. Green is `Rx(φ)` and the Y axis. Coral is the X axis.

## Key decisions

1. **Exact TypeScript statevector, not a Python quantum stack.** One and two qubits do not need PennyLane or TorchQuantum. Gradients and pictures stay in one process and update on a slider. A Python sidecar is only justified for a recorded many-qubit training run.
2. **Bloch ball first, satellite chart later.** The requested picture is the state trajectory. VIOLET's satellite chart explains basis amplitudes and angles in a 2D glyph. Both can sit side by side later. Cloning that glyph now would hide the geometry this project is for.
3. **⟨Z⟩ regression on a fixed 1D set.** A binary label in `{−1, +1}` makes the measurement a signed axis on the same sphere the user is watching. Six points are enough to see a valley and a floor: angle encoding plus two rotations cannot put every `x` on the correct pole, so the loss does not reach zero.
4. **Separate the inspected input from the training batch.** Watching one trajectory and descending the batch loss are different operations. Mixing them would make the sphere jump between dataset rows during training.
5. **Entanglement is a second mode with its own circuit.** Stuffing a CNOT into the one-qubit trainer would make ⟨Z⟩ a reduced expectation and would change the loss. The entangle mode exists to show the arrow leaving the surface, which a pure one-qubit state never does.
6. **Sample the unitary, do not rotate a Three.js arrow by hand.** A hand-rotated arrow drifts from the amplitudes as soon as two axes are composed. The mesh only displays coordinates the simulator computed.

## Out of scope for the base

- VIOLET's satellite chart and augmented heatmap
- Noise, tomography, and hardware execution
- Amplitude encoding, ZZ feature maps, and data re-uploading
- More than two qubits
- A 3D loss volume

## Open questions

- Should the first multi-qubit encoder be angle encoding on two features, or a ZZ feature map whose entanglement is part of the data layer?
- Should a later training run stream parameters from TorchQuantum, matching VIOLET's case studies, or stay on the in-browser parameter-shift loop?
- Is the public demo a static build on GitHub Pages, or only the local Vite app?
