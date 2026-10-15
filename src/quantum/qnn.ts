import { apply, blochOf, KET0, probabilities, rx, ry, type Bloch, type Qubit } from "./qubit";

export type Sample = { x: number; y: 1 | -1 };

/** Two clusters on a line. Labels are the targets for ⟨Z⟩, not measurement bits. */
export const DATASET: readonly Sample[] = [
  { x: -1.2, y: 1 },
  { x: -0.75, y: 1 },
  { x: -0.35, y: 1 },
  { x: 0.35, y: -1 },
  { x: 0.75, y: -1 },
  { x: 1.2, y: -1 },
];

export const INITIAL_THETA = 0.45;
export const INITIAL_PHI = 1.05;

export type StageName = "encode" | "ry" | "rx";

export type TrailSegment = {
  name: StageName;
  label: string;
  color: string;
  points: Bloch[];
};

const STAGE_META: Record<StageName, { label: string; color: string }> = {
  encode: { label: "Ry(x)", color: "#7eb6ff" },
  ry: { label: "Ry(θ)", color: "#e2b15a" },
  rx: { label: "Rx(φ)", color: "#8fbf9f" },
};

function rotate(name: StageName, angle: number, psi: Qubit): Qubit {
  const gate = name === "rx" ? rx(angle) : ry(angle);
  return apply(gate, psi);
}

/** Full circuit state for one classical input. */
export function runCircuit(x: number, theta: number, phi: number): Qubit {
  let psi = KET0;
  psi = rotate("encode", x, psi);
  psi = rotate("ry", theta, psi);
  psi = rotate("rx", phi, psi);
  return psi;
}

export function expectationZ(x: number, theta: number, phi: number): number {
  return blochOf(runCircuit(x, theta, phi)).z;
}

export function loss(theta: number, phi: number, data: readonly Sample[] = DATASET): number {
  let total = 0;
  for (const sample of data) {
    const error = expectationZ(sample.x, theta, phi) - sample.y;
    total += error * error;
  }
  return total / data.length;
}

export function accuracy(theta: number, phi: number, data: readonly Sample[] = DATASET): number {
  let correct = 0;
  for (const sample of data) {
    const predicted = expectationZ(sample.x, theta, phi) >= 0 ? 1 : -1;
    if (predicted === sample.y) correct += 1;
  }
  return correct / data.length;
}

/**
 * Gradient of the batch loss. Parameter shift is exact for Rx and Ry,
 * which are exp(−i α P / 2) for a Pauli P.
 */
export function gradients(
  theta: number,
  phi: number,
  data: readonly Sample[] = DATASET,
): { gTheta: number; gPhi: number } {
  const shift = Math.PI / 2;
  let gTheta = 0;
  let gPhi = 0;
  const scale = 2 / data.length;
  for (const sample of data) {
    const z = expectationZ(sample.x, theta, phi);
    const dZdTheta =
      (expectationZ(sample.x, theta + shift, phi) - expectationZ(sample.x, theta - shift, phi)) / 2;
    const dZdPhi =
      (expectationZ(sample.x, theta, phi + shift) - expectationZ(sample.x, theta, phi - shift)) / 2;
    const dLdZ = scale * (z - sample.y);
    gTheta += dLdZ * dZdTheta;
    gPhi += dLdZ * dZdPhi;
  }
  return { gTheta, gPhi };
}

export function descend(
  theta: number,
  phi: number,
  learningRate: number,
  data: readonly Sample[] = DATASET,
): { theta: number; phi: number; loss: number } {
  const g = gradients(theta, phi, data);
  const nextTheta = theta - learningRate * g.gTheta;
  const nextPhi = phi - learningRate * g.gPhi;
  return { theta: nextTheta, phi: nextPhi, loss: loss(nextTheta, nextPhi, data) };
}

export type LossGrid = {
  n: number;
  values: Float64Array;
  min: number;
  max: number;
};

/** Row-major grid. Column is θ, row is φ, both spanning [−π, π] with +φ at row 0. */
export function lossGrid(n = 48, data: readonly Sample[] = DATASET): LossGrid {
  const values = new Float64Array(n * n);
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let row = 0; row < n; row += 1) {
    const phi = Math.PI - (row / (n - 1)) * 2 * Math.PI;
    for (let col = 0; col < n; col += 1) {
      const theta = -Math.PI + (col / (n - 1)) * 2 * Math.PI;
      const value = loss(theta, phi, data);
      values[row * n + col] = value;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return { n, values, min, max };
}

const SAMPLES_PER_STAGE = 28;

/**
 * Bloch samples along the circuit. `reveal` is in [0, 3]: 1 means the encoder
 * has finished, 1.5 means the Ry(θ) stage is half applied.
 */
export function traceCircuit(
  x: number,
  theta: number,
  phi: number,
  reveal: number,
): { segments: TrailSegment[]; state: Qubit } {
  const stages: { name: StageName; angle: number }[] = [
    { name: "encode", angle: x },
    { name: "ry", angle: theta },
    { name: "rx", angle: phi },
  ];
  let psi: Qubit = KET0;
  const segments: TrailSegment[] = [];
  const clamped = Math.max(0, Math.min(3, reveal));

  stages.forEach((stage, index) => {
    const fraction = Math.max(0, Math.min(1, clamped - index));
    const meta = STAGE_META[stage.name];
    const points: Bloch[] = [blochOf(psi)];
    if (fraction > 0) {
      const steps = Math.max(2, Math.round(SAMPLES_PER_STAGE * fraction));
      for (let step = 1; step <= steps; step += 1) {
        const t = (step / steps) * fraction;
        points.push(blochOf(rotate(stage.name, stage.angle * t, psi)));
      }
      psi = rotate(stage.name, stage.angle * fraction, psi);
    }
    segments.push({ name: stage.name, label: meta.label, color: meta.color, points });
  });

  return { segments, state: psi };
}

export function measurementOf(psi: Qubit): { p0: number; p1: number; z: number } {
  const { p0, p1 } = probabilities(psi);
  return { p0, p1, z: blochOf(psi).z };
}
