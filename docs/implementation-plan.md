# Implementation plan

Each phase is a reviewable increment on `main`. The base that lands with this plan is Phase 0 plus the parts of Phase 1 that the readout already needs.

## End-to-end trunk

Specified in [end-to-end.md](end-to-end.md), before the code that implements it. Two data-reuploading layers, a classical head `ŷ = a⟨Z⟩ + c`, and the Bloch sphere as a probe on `|ψ⟩` after each gate. The loss heatmap becomes a `(b₁, φ₁)` slice of that eight-parameter loss.

## Phase 0 — Simulator and Bloch tracker

Status: done in this base.

- Exact single-qubit `Rx`, `Ry`, `Rz`, Bloch vector, and probabilities.
- Angle-encoding circuit `Ry(x)`, `Ry(θ)`, `Rx(φ)` with ⟨Z⟩.
- Trajectory sampled from partial gates, coloured by stage.
- Parameter-shift gradient, mean-squared loss, one gradient step, and a short run.
- Loss heatmap on `(θ, φ)` with the step path.
- Two-qubit `Ry`, controlled-`Ry`, reduced Bloch vectors, purity, and concurrence.
- Tests for the known states `|0⟩`, `|+⟩`, `|−i⟩`, the Bell point, the radius identity `|r| = √(1 − C²)`, parameter shift against a finite difference, and a downhill training step.
- Vite + React + react-three-fiber dashboard.

Files: `src/quantum/*`, `src/view/coords.ts`, `src/components/*`, `src/App.tsx`.

## Phase 1 — Read the circuit the way VIOLET asks

- Basis-state bar for the one-qubit amplitudes that stays in sync with playback (R1, R4). The base already shows P(0) and P(1).
- A gate strip that highlights the stage currently being revealed (R3). The base already has this strip.
- Satellite-style glyph per variational angle: arc for the angle, tick for the current value, and the basis probabilities beside it (R2). This is the first piece that is actually VIOLET's chart rather than the Bloch ball.
- Drag a point on the loss map to set `(θ, φ)` without taking a gradient step.

## Phase 2 — Ansatz over iterations

- Store a snapshot per gradient step: parameters, per-point ⟨Z⟩, and the Bloch coordinate of the inspected input.
- Scrub the iteration. The sphere shows that snapshot's trajectory. The loss sparkline highlights the same step (R2, R3, R6).
- Accuracy beside loss: count of training points whose sign(⟨Z⟩) matches the label.

## Phase 3 — Feature view

- Augmented heatmap: rows are training points, columns are measurement outcomes or ⟨Z⟩ bins, colour is probability (R5).
- Link a heatmap cell back to the Bloch point of that input at the selected iteration.
- Optional second classical feature, still one qubit, by data re-uploading: `Ry(x₁)`, ansatz, `Ry(x₂)`, ansatz.

## Phase 4 — Two-qubit encoder

Decide the open question in `docs/design.md` first.

- Two-feature angle encoding, one feature per qubit, then a small entangling ansatz.
- Two reduced Bloch balls plus a four-bar basis histogram.
- Loss from a two-qubit observable, recorded explicitly so it is not confused with a single-qubit ⟨Z⟩.

## Phase 5 — Landscape beyond two parameters

- Add `Rz(ω)` and show the loss map at a fixed `ω`, with a slider for the slice.
- A 3D surface is only a view onto that same grid. It does not replace the heatmap.

## Test bar for every phase

- New quantum behaviour gets a numeric test in `src/quantum`.
- A change to the scene frame gets a test in `src/view/coords.test.ts`.
- `npm test` and `npm run build` pass before a phase is pushed.
- The affected dashboard path is exercised in a browser: slider, playback, one training step, and the entangle control when that mode changed.

## Suggested pull requests

1. **Phase 0 base.** This tree. Simulator, sphere, loss map, entanglement mode, design notes.
2. **Iteration scrubber.** Phase 2 on top of the existing history list. Depends on PR 1.
3. **Satellite glyph and loss-map picking.** Phase 1. Depends on PR 1. Independent of PR 2.
4. **Feature heatmap.** Phase 3. Depends on PR 2, because a cell is tied to an iteration.
5. **Two-qubit encoder.** Phase 4. Depends on PR 1 only.
6. **Third-parameter slice.** Phase 5. Depends on PR 1.
