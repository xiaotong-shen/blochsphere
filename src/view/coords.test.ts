import { describe, expect, it } from "vitest";
import { toThree, wrapPi } from "./coords";

function cross(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

describe("Bloch to Three.js frame", () => {
  it("preserves orientation, with Bloch Z up", () => {
    const x = toThree({ x: 1, y: 0, z: 0 });
    const y = toThree({ x: 0, y: 1, z: 0 });
    const z = toThree({ x: 0, y: 0, z: 1 });
    expect(z).toEqual([0, 1, 0]);
    expect(cross(x, y).map((value) => (value === 0 ? 0 : value))).toEqual(z);
    expect(cross(y, z).map((value) => (value === 0 ? 0 : value))).toEqual(x);
    expect(cross(z, x).map((value) => (value === 0 ? 0 : value))).toEqual(y);
  });

  it("wraps angles into [−π, π)", () => {
    expect(wrapPi(Math.PI)).toBeCloseTo(-Math.PI);
    expect(wrapPi(-Math.PI)).toBeCloseTo(-Math.PI);
    expect(wrapPi(Math.PI + 0.2)).toBeCloseTo(-Math.PI + 0.2);
    expect(wrapPi(-3 * Math.PI)).toBeCloseTo(-Math.PI);
  });
});
