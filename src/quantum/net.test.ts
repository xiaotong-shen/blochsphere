import { describe, expect, it } from "vitest";
import { abs2 } from "./complex";
import {
  INITIAL_NET,
  accuracy,
  descend,
  expectationZ,
  gradients,
  loss,
  predict,
  quantumState,
  traceNet,
  type NetParams,
} from "./net";

const KEYS = ["w1", "b1", "phi1", "w2", "b2", "phi2", "a", "c"] as const;

describe("hybrid trunk", () => {
  it("keeps the classical head off the Bloch vector", () => {
    const inspected = 0.4;
    const base = quantumState(inspected, INITIAL_NET);
    const shifted = quantumState(inspected, { ...INITIAL_NET, a: -1.5, c: 0.7 });
    expect(base[0].re).toBeCloseTo(shifted[0].re);
    expect(base[1].im).toBeCloseTo(shifted[1].im);
    expect(predict(inspected, INITIAL_NET)).not.toBeCloseTo(predict(inspected, { ...INITIAL_NET, a: -1.5, c: 0.7 }));
  });

  it("ends the traced path on the measured ⟨Z⟩", () => {
    const traced = traceNet(-0.75, INITIAL_NET, 4);
    const tip = traced.segments[3].points.at(-1)!;
    expect(tip.z).toBeCloseTo(expectationZ(-0.75, INITIAL_NET));
    expect(abs2(traced.state[0]) + abs2(traced.state[1])).toBeCloseTo(1);
    expect(traceNet(-0.75, INITIAL_NET, 0).state[0].re).toBeCloseTo(1);
  });

  it("matches a finite difference on every parameter", () => {
    const grad = gradients(INITIAL_NET);
    const eps = 1e-4;
    for (const key of KEYS) {
      const plus: NetParams = { ...INITIAL_NET, [key]: INITIAL_NET[key] + eps };
      const minus: NetParams = { ...INITIAL_NET, [key]: INITIAL_NET[key] - eps };
      const finite = (loss(plus) - loss(minus)) / (2 * eps);
      expect(grad[key]).toBeCloseTo(finite, 3);
    }
  });

  it("descends the batch loss", () => {
    const before = loss(INITIAL_NET);
    let params = INITIAL_NET;
    for (let step = 0; step < 40; step += 1) {
      params = descend(params, 0.35).params;
    }
    expect(loss(params)).toBeLessThan(before * 0.55);
    expect(accuracy(params)).toBeGreaterThanOrEqual(accuracy(INITIAL_NET));
  });
});
