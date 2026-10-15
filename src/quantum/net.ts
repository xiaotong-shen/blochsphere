import { apply, blochOf, KET0, rx, ry, type Bloch, type Qubit } from "./qubit";
import { DATASET, type LossGrid, type Sample, type TrailSegment } from "./qnn";

/** Two data-reuploading layers plus a classical head. See docs/end-to-end.md. */
export type NetParams = {
  w1: number;
  b1: number;
  phi1: number;
  w2: number;
  b2: number;
  phi2: number;
  a: number;
  c: number;
};

export const INITIAL_NET: NetParams = {
  w1: 0.85,
  b1: 0.35,
  phi1: 0.9,
  w2: 0.45,
  b2: -0.25,
  phi2: 0.55,
  a: 1,
  c: 0,
};

export const NET_STAGE_COUNT = 4;

type GateStep = {
  id: string;
  label: string;
  color: string;
  axis: "x" | "y";
  angle: number;
};

function stepsFor(x: number, params: NetParams): GateStep[] {
  return [
    { id: "l1-ry", label: "L1 Ry", color: "#7eb6ff", axis: "y", angle: params.w1 * x + params.b1 },
    { id: "l1-rx", label: "L1 Rx", color: "#e2b15a", axis: "x", angle: params.phi1 },
    { id: "l2-ry", label: "L2 Ry", color: "#9ec9ff", axis: "y", angle: params.w2 * x + params.b2 },
    { id: "l2-rx", label: "L2 Rx", color: "#8fbf9f", axis: "x", angle: params.phi2 },
  ];
}

function rotate(axis: "x" | "y", angle: number, psi: Qubit): Qubit {
  return apply(axis === "x" ? rx(angle) : ry(angle), psi);
}

/** Quantum trunk only. The head parameters a and c are not read. */
export function quantumState(x: number, params: NetParams): Qubit {
  let psi = KET0;
  for (const step of stepsFor(x, params)) psi = rotate(step.axis, step.angle, psi);
  return psi;
}

export function expectationZ(x: number, params: NetParams): number {
  return blochOf(quantumState(x, params)).z;
}

export function predict(x: number, params: NetParams): number {
  return params.a * expectationZ(x, params) + params.c;
}

export function loss(params: NetParams, data: readonly Sample[] = DATASET): number {
  let total = 0;
  for (const sample of data) {
    const error = predict(sample.x, params) - sample.y;
    total += error * error;
  }
  return total / data.length;
}

export function accuracy(params: NetParams, data: readonly Sample[] = DATASET): number {
  let correct = 0;
  for (const sample of data) {
    const predicted = predict(sample.x, params) >= 0 ? 1 : -1;
    if (predicted === sample.y) correct += 1;
  }
  return correct / data.length;
}

export type NetGradient = Record<keyof NetParams, number>;

function shiftedZ(x: number, params: NetParams, key: "b1" | "phi1" | "b2" | "phi2", delta: number): number {
  return expectationZ(x, { ...params, [key]: params[key] + delta });
}

/** Batch gradient. Quantum angles use parameter shift; the head is analytic. */
export function gradients(params: NetParams, data: readonly Sample[] = DATASET): NetGradient {
  const shift = Math.PI / 2;
  const grad: NetGradient = { w1: 0, b1: 0, phi1: 0, w2: 0, b2: 0, phi2: 0, a: 0, c: 0 };
  const scale = 2 / data.length;

  for (const sample of data) {
    const z = expectationZ(sample.x, params);
    const yHat = params.a * z + params.c;
    const dLossDy = scale * (yHat - sample.y);
    const dLossDz = dLossDy * params.a;

    const dzDb1 = (shiftedZ(sample.x, params, "b1", shift) - shiftedZ(sample.x, params, "b1", -shift)) / 2;
    const dzDphi1 =
      (shiftedZ(sample.x, params, "phi1", shift) - shiftedZ(sample.x, params, "phi1", -shift)) / 2;
    const dzDb2 = (shiftedZ(sample.x, params, "b2", shift) - shiftedZ(sample.x, params, "b2", -shift)) / 2;
    const dzDphi2 =
      (shiftedZ(sample.x, params, "phi2", shift) - shiftedZ(sample.x, params, "phi2", -shift)) / 2;

    grad.w1 += dLossDz * dzDb1 * sample.x;
    grad.b1 += dLossDz * dzDb1;
    grad.phi1 += dLossDz * dzDphi1;
    grad.w2 += dLossDz * dzDb2 * sample.x;
    grad.b2 += dLossDz * dzDb2;
    grad.phi2 += dLossDz * dzDphi2;
    grad.a += dLossDy * z;
    grad.c += dLossDy;
  }

  return grad;
}

export function descend(
  params: NetParams,
  learningRate: number,
  data: readonly Sample[] = DATASET,
): { params: NetParams; loss: number } {
  const grad = gradients(params, data);
  const next: NetParams = { ...params };
  (Object.keys(grad) as (keyof NetParams)[]).forEach((key) => {
    next[key] = params[key] - learningRate * grad[key];
  });
  return { params: next, loss: loss(next, data) };
}

/**
 * Loss slice L(b₁, φ₁). Every other parameter stays at `params`.
 * Columns run b₁ from −π to π. Rows run φ₁ from π down to −π.
 */
export function sliceGrid(params: NetParams, n = 32, data: readonly Sample[] = DATASET): LossGrid {
  const values = new Float64Array(n * n);
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let row = 0; row < n; row += 1) {
    const phi1 = Math.PI - (row / (n - 1)) * 2 * Math.PI;
    for (let col = 0; col < n; col += 1) {
      const b1 = -Math.PI + (col / (n - 1)) * 2 * Math.PI;
      const value = loss({ ...params, b1, phi1 }, data);
      values[row * n + col] = value;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return { n, values, min, max };
}

const SAMPLES_PER_STAGE = 24;

/** `reveal` is in [0, 4]. Each unit is one gate of the trunk, in circuit order. */
export function traceNet(
  x: number,
  params: NetParams,
  reveal: number,
): { segments: TrailSegment[]; state: Qubit } {
  const steps = stepsFor(x, params);
  let psi: Qubit = KET0;
  const segments: TrailSegment[] = [];
  const clamped = Math.max(0, Math.min(NET_STAGE_COUNT, reveal));

  steps.forEach((step, index) => {
    const fraction = Math.max(0, Math.min(1, clamped - index));
    const points: Bloch[] = [blochOf(psi)];
    if (fraction > 0) {
      const count = Math.max(2, Math.round(SAMPLES_PER_STAGE * fraction));
      for (let sample = 1; sample <= count; sample += 1) {
        const t = (sample / count) * fraction;
        points.push(blochOf(rotate(step.axis, step.angle * t, psi)));
      }
      psi = rotate(step.axis, step.angle * fraction, psi);
    }
    segments.push({ id: step.id, label: step.label, color: step.color, points });
  });

  return { segments, state: psi };
}
