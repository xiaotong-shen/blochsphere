import { describe, expect, it } from "vitest";
import { abs2 } from "./complex";
import {
  accuracy,
  DATASET,
  descend,
  expectationZ,
  gradients,
  loss,
  traceCircuit,
} from "./qnn";
import { apply, blochOf, KET0, norm2, radius, rx, ry } from "./qubit";
import { concurrence, entangleCircuit, pairNorm2, reduceQubit0, reduceQubit1, traceEntangle } from "./two";

const SQRT2 = Math.SQRT1_2;

describe("single-qubit Bloch vectors", () => {
  it("places the known poles and equator states", () => {
    expect(blochOf(KET0)).toMatchObject({ x: 0, y: 0, z: 1 });

    const plus = apply(ry(Math.PI / 2), KET0);
    const plusBloch = blochOf(plus);
    expect(plusBloch.x).toBeCloseTo(1);
    expect(plusBloch.y).toBeCloseTo(0);
    expect(plusBloch.z).toBeCloseTo(0);

    const minusI = apply(rx(Math.PI / 2), KET0);
    const minusIBloch = blochOf(minusI);
    expect(minusIBloch.x).toBeCloseTo(0);
    expect(minusIBloch.y).toBeCloseTo(-1);
    expect(minusIBloch.z).toBeCloseTo(0);

    const one = apply(ry(Math.PI), KET0);
    expect(blochOf(one).z).toBeCloseTo(-1);
    expect(norm2(one)).toBeCloseTo(1);
  });

  it("matches ⟨Z⟩ at the end of the sampled trajectory", () => {
    const x = -0.75;
    const theta = 0.45;
    const phi = 1.05;
    const traced = traceCircuit(x, theta, phi, 3);
    const tip = traced.segments[2].points.at(-1)!;
    expect(tip.z).toBeCloseTo(expectationZ(x, theta, phi));
    expect(abs2(traced.state[0]) + abs2(traced.state[1])).toBeCloseTo(1);
  });
});

describe("training", () => {
  it("matches a finite difference of the batch loss", () => {
    const theta = 0.45;
    const phi = 1.05;
    const g = gradients(theta, phi);
    const eps = 1e-4;
    const dTheta = (loss(theta + eps, phi) - loss(theta - eps, phi)) / (2 * eps);
    const dPhi = (loss(theta, phi + eps) - loss(theta, phi - eps)) / (2 * eps);
    expect(g.gTheta).toBeCloseTo(dTheta, 3);
    expect(g.gPhi).toBeCloseTo(dPhi, 3);
  });

  it("takes a downhill step and separates the two clusters", () => {
    const before = loss(0.45, 1.05);
    let theta = 0.45;
    let phi = 1.05;
    for (let step = 0; step < 40; step += 1) {
      const next = descend(theta, phi, 0.45);
      theta = next.theta;
      phi = next.phi;
    }
    expect(loss(theta, phi)).toBeLessThan(before * 0.5);
    expect(accuracy(theta, phi)).toBe(1);
    expect(DATASET).toHaveLength(6);
  });
});

describe("entanglement", () => {
  it("keeps a product state on the sphere and a Bell state at the origin", () => {
    const product = entangleCircuit(0.8, -0.4, 0);
    expect(pairNorm2(product)).toBeCloseTo(1);
    expect(concurrence(product)).toBeCloseTo(0);
    expect(radius(reduceQubit0(product).bloch)).toBeCloseTo(1);
    expect(radius(reduceQubit1(product).bloch)).toBeCloseTo(1);
    expect(reduceQubit0(product).purity).toBeCloseTo(1);

    const bell = entangleCircuit(Math.PI / 2, 0, Math.PI);
    expect(concurrence(bell)).toBeCloseTo(1);
    expect(radius(reduceQubit0(bell).bloch)).toBeCloseTo(0, 5);
    expect(radius(reduceQubit1(bell).bloch)).toBeCloseTo(0, 5);
    expect(reduceQubit0(bell).purity).toBeCloseTo(0.5);
    expect(Math.hypot(bell[0].re, bell[0].im)).toBeCloseTo(SQRT2);
    expect(Math.hypot(bell[3].re, bell[3].im)).toBeCloseTo(SQRT2);
  });

  it("ends the entangle trace on the same state as the circuit", () => {
    const traced = traceEntangle(0.7, -0.4, 1.2);
    const direct = entangleCircuit(0.7, -0.4, 1.2);
    traced.final.forEach((amplitude, index) => {
      expect(amplitude.re).toBeCloseTo(direct[index].re);
      expect(amplitude.im).toBeCloseTo(direct[index].im);
    });
    expect(traced.q0Pull.at(-1)?.z).toBeCloseTo(reduceQubit0(direct).bloch.z);
  });

  it("shortens the Bloch vector as √(1 − C²)", () => {
    const psi = entangleCircuit(1.1, -0.6, 1.4);
    const c = concurrence(psi);
    const r0 = radius(reduceQubit0(psi).bloch);
    const r1 = radius(reduceQubit1(psi).bloch);
    expect(r0).toBeCloseTo(Math.sqrt(1 - c * c));
    expect(r1).toBeCloseTo(Math.sqrt(1 - c * c));
  });
});
