import { abs2, add, c, conj, hypot, mul, scale, sub, type C } from "./complex";
import { ry, type Bloch, type Gate } from "./qubit";

/**
 * Two-qubit amplitudes with qubit 0 as the least significant bit:
 * [|00⟩, |01⟩, |10⟩, |11⟩].
 */
export type Pair = [C, C, C, C];

export const KET00: Pair = [c(1), c(0), c(0), c(0)];

function mix(psi: Pair, i: number, j: number, gate: Gate): Pair {
  const next = [...psi] as Pair;
  const a = psi[i];
  const b = psi[j];
  next[i] = add(mul(gate[0], a), mul(gate[1], b));
  next[j] = add(mul(gate[2], a), mul(gate[3], b));
  return next;
}

/** Ry on qubit 0, the least significant bit. */
export function ryQubit0(angle: number, psi: Pair): Pair {
  const gate = ry(angle);
  return mix(mix(psi, 0, 1, gate), 2, 3, gate);
}

/** Ry on qubit 1. */
export function ryQubit1(angle: number, psi: Pair): Pair {
  const gate = ry(angle);
  return mix(mix(psi, 0, 2, gate), 1, 3, gate);
}

/**
 * Controlled-Ry. Qubit 0 is the control, qubit 1 is the target.
 * γ = π and a |+⟩ control produces a Bell state from |00⟩.
 */
export function cryControl0(gamma: number, psi: Pair): Pair {
  const half = gamma / 2;
  const cos = Math.cos(half);
  const sin = Math.sin(half);
  const c1 = psi[1];
  const c3 = psi[3];
  const next = [...psi] as Pair;
  next[1] = add(scale(c1, cos), scale(c3, -sin));
  next[3] = add(scale(c1, sin), scale(c3, cos));
  return next;
}

export type Reduced = {
  bloch: Bloch;
  purity: number;
};

function reduced(rho00: number, rho11: number, rho01: C): Reduced {
  const bloch = {
    x: 2 * rho01.re,
    y: -2 * rho01.im,
    z: rho00 - rho11,
  };
  const purity = rho00 * rho00 + rho11 * rho11 + 2 * abs2(rho01);
  return { bloch, purity };
}

/** Partial trace over qubit 1. */
export function reduceQubit0(psi: Pair): Reduced {
  const rho00 = abs2(psi[0]) + abs2(psi[2]);
  const rho11 = abs2(psi[1]) + abs2(psi[3]);
  const rho01 = add(mul(psi[0], conj(psi[1])), mul(psi[2], conj(psi[3])));
  return reduced(rho00, rho11, rho01);
}

/** Partial trace over qubit 0. */
export function reduceQubit1(psi: Pair): Reduced {
  const rho00 = abs2(psi[0]) + abs2(psi[1]);
  const rho11 = abs2(psi[2]) + abs2(psi[3]);
  const rho01 = add(mul(psi[0], conj(psi[2])), mul(psi[1], conj(psi[3])));
  return reduced(rho00, rho11, rho01);
}

/** Concurrence of a pure two-qubit state: 2 |c00 c11 − c01 c10|. */
export function concurrence(psi: Pair): number {
  const det = sub(mul(psi[0], psi[3]), mul(psi[1], psi[2]));
  return 2 * hypot(det);
}

export function pairNorm2(psi: Pair): number {
  return psi.reduce((sum, amp) => sum + abs2(amp), 0);
}

/** Product rotations, then a controlled-Ry. Both qubits start in |0⟩. */
export function entangleCircuit(theta0: number, theta1: number, gamma: number): Pair {
  let psi = KET00;
  psi = ryQubit0(theta0, psi);
  psi = ryQubit1(theta1, psi);
  psi = cryControl0(gamma, psi);
  return psi;
}

export type EntangleTrace = {
  q0Local: Bloch[];
  q0Pull: Bloch[];
  q1Local: Bloch[];
  q1Pull: Bloch[];
  final: Pair;
};

function pushReduced(psi: Pair, q0: Bloch[], q1: Bloch[]) {
  q0.push(reduceQubit0(psi).bloch);
  q1.push(reduceQubit1(psi).bloch);
}

/** Local rotations, then the inward path as the controlled rotation turns on. */
export function traceEntangle(theta0: number, theta1: number, gamma: number): EntangleTrace {
  const steps = 24;
  const q0Local: Bloch[] = [];
  const q1Local: Bloch[] = [];
  const q0Pull: Bloch[] = [];
  const q1Pull: Bloch[] = [];

  pushReduced(KET00, q0Local, q1Local);
  for (let step = 1; step <= steps; step += 1) {
    pushReduced(ryQubit0((theta0 * step) / steps, KET00), q0Local, q1Local);
  }
  const after0 = ryQubit0(theta0, KET00);
  for (let step = 1; step <= steps; step += 1) {
    pushReduced(ryQubit1((theta1 * step) / steps, after0), q0Local, q1Local);
  }
  const afterLocal = ryQubit1(theta1, after0);
  pushReduced(afterLocal, q0Pull, q1Pull);
  for (let step = 1; step <= steps; step += 1) {
    pushReduced(cryControl0((gamma * step) / steps, afterLocal), q0Pull, q1Pull);
  }

  return {
    q0Local,
    q0Pull,
    q1Local,
    q1Pull,
    final: cryControl0(gamma, afterLocal),
  };
}
