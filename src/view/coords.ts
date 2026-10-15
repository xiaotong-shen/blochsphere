import type { Bloch } from "../quantum/qubit";

/**
 * Bloch (x, y, z) is right-handed and Z-up.
 * Three.js is right-handed and Y-up, with +Z toward the default camera.
 * (x, y, z) → (x, z, −y) sends Bloch Z up and Bloch Y into the screen.
 */
function signed(value: number): number {
  return value === 0 ? 0 : value;
}

export function toThree(b: Bloch): [number, number, number] {
  return [signed(b.x), signed(b.z), signed(-b.y)];
}

export function wrapPi(angle: number): number {
  const turn = Math.PI * 2;
  return angle - Math.floor((angle + Math.PI) / turn) * turn;
}
