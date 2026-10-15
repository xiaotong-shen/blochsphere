import { abs2, add, c, conj, mul, type C } from "./complex";

/** Row-major 2×2 unitary. */
export type Gate = [C, C, C, C];

/** Amplitudes [α, β] of α|0⟩ + β|1⟩. */
export type Qubit = [C, C];

export type Bloch = { x: number; y: number; z: number };

export const KET0: Qubit = [c(1), c(0)];

export function rx(angle: number): Gate {
  const half = angle / 2;
  const cos = Math.cos(half);
  const sin = Math.sin(half);
  return [c(cos), c(0, -sin), c(0, -sin), c(cos)];
}

export function ry(angle: number): Gate {
  const half = angle / 2;
  const cos = Math.cos(half);
  const sin = Math.sin(half);
  return [c(cos), c(-sin), c(sin), c(cos)];
}

export function rz(angle: number): Gate {
  const half = angle / 2;
  return [c(Math.cos(-half), Math.sin(-half)), c(0), c(0), c(Math.cos(half), Math.sin(half))];
}

export function apply(gate: Gate, psi: Qubit): Qubit {
  const [a, b] = psi;
  return [
    add(mul(gate[0], a), mul(gate[1], b)),
    add(mul(gate[2], a), mul(gate[3], b)),
  ];
}

/** Pauli expectations. North pole +Z is |0⟩. */
export function blochOf(psi: Qubit): Bloch {
  const [alpha, beta] = psi;
  const product = mul(conj(alpha), beta);
  return {
    x: 2 * product.re,
    y: 2 * product.im,
    z: abs2(alpha) - abs2(beta),
  };
}

export function norm2(psi: Qubit): number {
  return abs2(psi[0]) + abs2(psi[1]);
}

export function probabilities(psi: Qubit): { p0: number; p1: number } {
  return { p0: abs2(psi[0]), p1: abs2(psi[1]) };
}

export function radius(b: Bloch): number {
  return Math.hypot(b.x, b.y, b.z);
}
