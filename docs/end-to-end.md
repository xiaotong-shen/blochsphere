# End-to-end net, and where the Bloch sphere sits

The sphere is not the model. It is a probe on the hidden quantum state, in the same place an activation tape sits in a classical net. This note fixes the net that the dashboard trains, and the exact slot the picture occupies. The implementation that follows this commit should match it.

## The net

One feature, one qubit, two data-reuploading layers, then a classical affine head. Every pure state of that qubit is one point on the Bloch sphere, so the whole hidden state of the quantum trunk is visible. A wider classical network, or a second qubit inside the trainable trunk, would leave part of the forward pass off the picture.

```
x
  → layer 1:  Ry(w₁ x + b₁)  then  Rx(φ₁)
  → layer 2:  Ry(w₂ x + b₂)  then  Rx(φ₂)
  → z = ⟨Z⟩
  → ŷ = a z + c
  → L = mean (ŷ − y)²
```

Each layer is the quantum analogue of one neuron: a data-dependent rotation, then a variational rotation. Uploading `x` again in layer 2 is data re-uploading. That is what lets one qubit represent a nonlinear function of one input. The head is ordinary. `a` and `c` rescale the measured scalar. They are trained in the same step as the angles.

The training set stays the six-point line from the base, labels in `{+1, −1}`. Accuracy is the fraction of points whose `sign(ŷ)` matches the label.

Parameters, eight of them, updated together:

| Symbol | Role |
| --- | --- |
| `w₁`, `b₁` | layer-1 angle `w₁ x + b₁` |
| `φ₁` | layer-1 variational `Rx` |
| `w₂`, `b₂` | layer-2 angle |
| `φ₂` | layer-2 variational `Rx` |
| `a`, `c` | classical head `ŷ = a ⟨Z⟩ + c` |

## Where the sphere sits

```
x → Ry(w₁x+b₁) → Rx(φ₁) → Ry(w₂x+b₂) → Rx(φ₂) → ⟨Z⟩ → a⟨Z⟩+c → loss
       │            │           │            │
       └──────── Bloch samples of |ψ⟩ ───────┘
                                         measurement ends the probe
```

After every gate the simulator records the Bloch vector of the inspected input. Playback draws those four arcs in order. The arrow is `|ψ⟩` at the revealed gate.

The probe stops at the measurement. `⟨Z⟩` is a number in `[−1, 1]`. Multiplying by `a` and adding `c` cannot move the arrow. Dragging the head changes `ŷ` and the loss, and the sphere stays still. That is the point of the split: the ball shows the quantum trunk, the readout shows the classical head.

The entanglement view stays a separate instrument. A controlled rotation inside this trunk would mix the reduced state and the arrow would leave the surface. That demonstration is not a layer of this net.

## Gradients

`Rx` and `Ry` are `exp(−i α P / 2)`. For a rotation angle `α`,

```
∂⟨Z⟩/∂α = (⟨Z⟩(α + π/2) − ⟨Z⟩(α − π/2)) / 2
```

Layer angles are affine in the input, `α = w x + b`, so

```
∂⟨Z⟩/∂w = (∂⟨Z⟩/∂α) x
∂⟨Z⟩/∂b = ∂⟨Z⟩/∂α
```

The head is differentiable in the ordinary way. With `ŷ = a z + c` and `L = mean (ŷ − y)²`,

```
∂L/∂a = mean 2 (ŷ − y) z
∂L/∂c = mean 2 (ŷ − y)
∂L/∂z = mean 2 (ŷ − y) a
```

Quantum parameter derivatives are `∂L/∂z` times `∂z/∂α`. One gradient step subtracts `η` times this eight-vector from the parameters. There is no second optimizer and no stop-gradient between the head and the trunk.

## What the picture can and cannot show

The loss depends on eight numbers. The heatmap stays two-dimensional. It is the slice `L(b₁, φ₁)` with `w₁`, layer 2, and the head held at their current values. The polyline is only the `(b₁, φ₁)` projection of the full training step. A step can leave that slice's value even when the drawn point barely moves, because another parameter carried the loss.

The inspected `x` still drives the sphere. The batch still drives the loss and the gradient. Those stay different jobs.

## Out of this integration

- A classical MLP in front of the encoder
- More than one trained qubit
- A learned measurement basis
- Replacing the slice with an 8D landscape
