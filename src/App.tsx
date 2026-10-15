import { useEffect, useMemo, useRef, useState } from "react";
import { BlochScene, type SphereDraw } from "./components/BlochScene";
import { LossMap } from "./components/LossMap";
import type { C } from "./quantum/complex";
import {
  accuracy,
  DATASET,
  descend,
  expectationZ,
  INITIAL_PHI,
  INITIAL_THETA,
  loss,
  lossGrid,
  traceCircuit,
  type TrailSegment,
} from "./quantum/qnn";
import { blochOf, radius } from "./quantum/qubit";
import { concurrence, reduceQubit0, reduceQubit1, traceEntangle } from "./quantum/two";
import { wrapPi } from "./view/coords";

type Mode = "qnn" | "entangle";
type Step = { theta: number; phi: number; loss: number };

const LEARNING_RATE = 0.4;
const TRAIN_STEPS = 28;

function fmt(value: number, digits = 3): string {
  const text = Math.abs(value).toFixed(digits);
  return value < 0 ? `−${text}` : text;
}

function fmtAmp(amplitude: C): string {
  if (Math.abs(amplitude.im) < 5e-4) return fmt(amplitude.re);
  const sign = amplitude.im < 0 ? "−" : "+";
  return `${fmt(amplitude.re)} ${sign} ${Math.abs(amplitude.im).toFixed(3)}i`;
}

export function App() {
  const [mode, setMode] = useState<Mode>("qnn");
  const [x, setX] = useState(-0.75);
  const [theta, setTheta] = useState(INITIAL_THETA);
  const [phi, setPhi] = useState(INITIAL_PHI);
  const [reveal, setReveal] = useState(3);
  const [playing, setPlaying] = useState(false);
  const [running, setRunning] = useState(false);
  const [theta0, setTheta0] = useState(Math.PI / 2);
  const [theta1, setTheta1] = useState(0);
  const [gamma, setGamma] = useState(0);
  const [history, setHistory] = useState<Step[]>(() => [
    { theta: INITIAL_THETA, phi: INITIAL_PHI, loss: loss(INITIAL_THETA, INITIAL_PHI) },
  ]);

  const grid = useMemo(() => lossGrid(46), []);
  const traced = useMemo(() => traceCircuit(x, theta, phi, reveal), [phi, reveal, theta, x]);
  const entangled = useMemo(() => traceEntangle(theta0, theta1, gamma), [gamma, theta0, theta1]);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const start = performance.now();
    const duration = 2800;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setReveal(t * 3);
      if (t < 1) frame = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const paramsRef = useRef({ theta, phi });
  paramsRef.current = { theta, phi };

  useEffect(() => {
    if (!running) return;
    let remaining = TRAIN_STEPS;
    const timer = window.setInterval(() => {
      const current = paramsRef.current;
      const next = descend(current.theta, current.phi, LEARNING_RATE);
      paramsRef.current = { theta: next.theta, phi: next.phi };
      setTheta(next.theta);
      setPhi(next.phi);
      setHistory((steps) => [...steps, next]);
      remaining -= 1;
      if (remaining <= 0) setRunning(false);
    }, 90);
    return () => window.clearInterval(timer);
  }, [running]);

  const currentLoss = loss(theta, phi);
  const currentAccuracy = accuracy(theta, phi);
  const bloch = blochOf(traced.state);
  const p0 = traced.state[0].re ** 2 + traced.state[0].im ** 2;
  const p1 = traced.state[1].re ** 2 + traced.state[1].im ** 2;
  const activeStage = Math.min(2, Math.floor(Math.max(0, reveal - 1e-6)));

  const q0 = reduceQubit0(entangled.final);
  const q1 = reduceQubit1(entangled.final);
  const entangleC = concurrence(entangled.final);
  const r0 = radius(q0.bloch);
  const r1 = radius(q1.bloch);

  const spheres: SphereDraw[] =
    mode === "qnn"
      ? [
          {
            id: "qubit",
            offset: [0, 0, 0],
            segments: traced.segments,
            tip: traced.segments[2].points.at(-1) ?? bloch,
            caption: "inspected qubit",
          },
        ]
      : [
          {
            id: "q0",
            offset: [-1.75, 0, 0],
            segments: entangleSegments("q0", entangled.q0Local, entangled.q0Pull),
            tip: q0.bloch,
            caption: "qubit 0",
          },
          {
            id: "q1",
            offset: [1.75, 0, 0],
            segments: entangleSegments("q1", entangled.q1Local, entangled.q1Pull),
            tip: q1.bloch,
            caption: "qubit 1",
          },
        ];

  function inspect(value: number) {
    setX(value);
    setPlaying(false);
    setReveal(3);
  }

  function play() {
    setRunning(false);
    setReveal(0);
    setPlaying(true);
  }

  function stepOnce() {
    setPlaying(false);
    setReveal(3);
    const next = descend(theta, phi, LEARNING_RATE);
    setTheta(next.theta);
    setPhi(next.phi);
    setHistory((steps) => [...steps, next]);
  }

  function resetTrain() {
    setRunning(false);
    setPlaying(false);
    setReveal(3);
    setTheta(INITIAL_THETA);
    setPhi(INITIAL_PHI);
    setHistory([{ theta: INITIAL_THETA, phi: INITIAL_PHI, loss: loss(INITIAL_THETA, INITIAL_PHI) }]);
  }

  return (
    <div className="app">
      <header>
        <div>
          <p className="eyebrow">QNN state space</p>
          <h1>blochsphere</h1>
        </div>
        <nav>
          <button type="button" className={mode === "qnn" ? "tab on" : "tab"} onClick={() => setMode("qnn")}>
            Encoder and ansatz
          </button>
          <button
            type="button"
            className={mode === "entangle" ? "tab on" : "tab"}
            onClick={() => setMode("entangle")}
          >
            Entanglement
          </button>
        </nav>
      </header>

      <main className="workspace">
        <aside className="panel controls">
          {mode === "qnn" ? (
            <>
              <p className="section-label">Classical input</p>
              <label>
                x = {fmt(x, 2)} rad
                <input
                  type="range"
                  min={-Math.PI}
                  max={Math.PI}
                  step={0.01}
                  value={x}
                  onChange={(event) => inspect(Number(event.target.value))}
                />
              </label>
              <div className="point-list">
                {DATASET.map((sample) => (
                  <button
                    key={sample.x}
                    type="button"
                    className={Math.abs(sample.x - x) < 1e-6 ? "point on" : "point"}
                    onClick={() => inspect(sample.x)}
                  >
                    <span>x {fmt(sample.x, 2)}</span>
                    <span className={sample.y > 0 ? "pos" : "neg"}>y {sample.y > 0 ? "+1" : "−1"}</span>
                    <span>{fmt(expectationZ(sample.x, theta, phi), 2)}</span>
                  </button>
                ))}
              </div>
              <p className="note">The sphere follows this input. Training still uses all six rows.</p>

              <p className="section-label">Ansatz</p>
              <label>
                θ = {fmt(wrapPi(theta), 2)}
                <input
                  type="range"
                  min={-Math.PI}
                  max={Math.PI}
                  step={0.01}
                  value={wrapPi(theta)}
                  onChange={(event) => {
                    setTheta(Number(event.target.value));
                    setPlaying(false);
                    setReveal(3);
                  }}
                />
              </label>
              <label>
                φ = {fmt(wrapPi(phi), 2)}
                <input
                  type="range"
                  min={-Math.PI}
                  max={Math.PI}
                  step={0.01}
                  value={wrapPi(phi)}
                  onChange={(event) => {
                    setPhi(Number(event.target.value));
                    setPlaying(false);
                    setReveal(3);
                  }}
                />
              </label>

              <div className="circuit" aria-label="Circuit">
                <span>|0⟩</span>
                <span className={activeStage === 0 && playing ? "gate hot" : "gate encode"}>Ry(x)</span>
                <span className={activeStage === 1 && playing ? "gate hot" : "gate ry"}>Ry(θ)</span>
                <span className={activeStage === 2 && playing ? "gate hot" : "gate rx"}>Rx(φ)</span>
                <span>⟨Z⟩</span>
              </div>

              <div className="button-row">
                <button type="button" onClick={play} disabled={playing}>
                  Play path
                </button>
                <button type="button" onClick={stepOnce} disabled={running}>
                  One step
                </button>
                <button type="button" onClick={() => setRunning((value) => !value)}>
                  {running ? "Stop" : "Train"}
                </button>
                <button type="button" onClick={resetTrain}>
                  Reset
                </button>
              </div>
            </>
          ) : (
            <>
              <p className="section-label">Product state, then a controlled rotation</p>
              <div className="circuit">
                <span>|00⟩</span>
                <span className="gate ry">Ry(θ₀)</span>
                <span className="gate ry">Ry(θ₁)</span>
                <span className="gate encode">CRy(γ)</span>
              </div>
              <label>
                θ₀ = {fmt(theta0, 2)}
                <input
                  type="range"
                  min={-Math.PI}
                  max={Math.PI}
                  step={0.01}
                  value={theta0}
                  onChange={(event) => setTheta0(Number(event.target.value))}
                />
              </label>
              <label>
                θ₁ = {fmt(theta1, 2)}
                <input
                  type="range"
                  min={-Math.PI}
                  max={Math.PI}
                  step={0.01}
                  value={theta1}
                  onChange={(event) => setTheta1(Number(event.target.value))}
                />
              </label>
              <label>
                γ = {fmt(gamma, 2)}
                <input
                  type="range"
                  min={0}
                  max={Math.PI}
                  step={0.01}
                  value={gamma}
                  onChange={(event) => setGamma(Number(event.target.value))}
                />
              </label>
              <div className="button-row">
                <button
                  type="button"
                  onClick={() => {
                    setTheta0(0.9);
                    setTheta1(-0.5);
                    setGamma(0);
                  }}
                >
                  Separable
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setTheta0(Math.PI / 2);
                    setTheta1(0);
                    setGamma(Math.PI / 2);
                  }}
                >
                  Partial
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setTheta0(Math.PI / 2);
                    setTheta1(0);
                    setGamma(Math.PI);
                  }}
                >
                  Bell
                </button>
              </div>
              <p className="note">
                CRy is controlled by qubit 0. Raising γ pulls both reduced states inside the ball. At θ₀ = π/2, θ₁ = 0
                and γ = π the joint state is the Bell state.
              </p>
            </>
          )}
        </aside>

        <section className="stage">
          <BlochScene spheres={spheres} />
        </section>

        <aside className="panel readout">
          {mode === "qnn" ? (
            <>
              <p className="section-label">State at the revealed gate</p>
              <p className="amplitude">
                {fmtAmp(traced.state[0])} |0⟩ + {fmtAmp(traced.state[1])} |1⟩
              </p>
              <dl>
                <div>
                  <dt>r</dt>
                  <dd>
                    ({fmt(bloch.x)}, {fmt(bloch.y)}, {fmt(bloch.z)})
                  </dd>
                </div>
                <div>
                  <dt>radius</dt>
                  <dd>{fmt(radius(bloch))}</dd>
                </div>
                <div>
                  <dt>P(0)</dt>
                  <dd>{fmt(p0)}</dd>
                </div>
                <div>
                  <dt>P(1)</dt>
                  <dd>{fmt(p1)}</dd>
                </div>
                <div>
                  <dt>⟨Z⟩</dt>
                  <dd>{fmt(bloch.z)}</dd>
                </div>
                <div>
                  <dt>P(0)+P(1)</dt>
                  <dd>{fmt(p0 + p1)}</dd>
                </div>
              </dl>
              <p className="section-label">Batch</p>
              <dl>
                <div>
                  <dt>loss</dt>
                  <dd>{fmt(currentLoss)}</dd>
                </div>
                <div>
                  <dt>sign accuracy</dt>
                  <dd>{fmt(currentAccuracy, 2)}</dd>
                </div>
              </dl>
              <Sparkline values={history.map((step) => step.loss)} />
            </>
          ) : (
            <>
              <p className="section-label">Reduced states</p>
              <dl>
                <div>
                  <dt>concurrence</dt>
                  <dd>{fmt(entangleC)}</dd>
                </div>
                <div>
                  <dt>qubit 0 radius</dt>
                  <dd>{fmt(r0)}</dd>
                </div>
                <div>
                  <dt>qubit 1 radius</dt>
                  <dd>{fmt(r1)}</dd>
                </div>
                <div>
                  <dt>√(1−C²)</dt>
                  <dd>{fmt(Math.sqrt(Math.max(0, 1 - entangleC * entangleC)))}</dd>
                </div>
                <div>
                  <dt>purity</dt>
                  <dd>{fmt(q0.purity)}</dd>
                </div>
              </dl>
              <p className="note">
                A pure two-qubit state has reduced radius √(1−C²). The arrow stays on the surface only while the qubits
                are separable.
              </p>
              <ul className="basis">
                {(["|00⟩", "|01⟩", "|10⟩", "|11⟩"] as const).map((label, index) => (
                  <li key={label}>
                    <span>{label}</span>
                    <span>{fmt(entangled.final[index].re ** 2 + entangled.final[index].im ** 2)}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </aside>
      </main>

      <footer className="panel landscape">
        {mode === "qnn" ? (
          <>
            <div className="landscape-copy">
              <p className="section-label">Loss on the training set</p>
              <p>
                L(θ, φ) = mean (⟨Z⟩ − y)². The pale path is gradient descent with step size {LEARNING_RATE}. Low loss is
                teal, high loss is sand. Dragging θ or φ moves the marker and does not record a step.
              </p>
              <p className="range">
                loss range {fmt(grid.min, 2)} – {fmt(grid.max, 2)}
              </p>
            </div>
            <div className="map-frame">
              <LossMap grid={grid} current={{ theta, phi }} history={history} />
            </div>
          </>
        ) : (
          <div className="landscape-copy wide">
            <p className="section-label">Why the arrow leaves the sphere</p>
            <p>
              Each ball is a reduced state, not the joint state. Local Ry gates move the points on the surface. The
              controlled rotation correlates the qubits, the partial trace becomes mixed, and both points slide toward
              the centre by the same amount.
            </p>
          </div>
        )}
      </footer>
    </div>
  );
}

function entangleSegments(id: string, local: SphereDraw["segments"][number]["points"], pull: typeof local): TrailSegment[] {
  return [
    { name: "encode", label: `${id} local`, color: "#e2b15a", points: local },
    { name: "rx", label: `${id} pull`, color: "#e07a5f", points: pull },
  ];
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const width = 240;
  const height = 48;
  const points = values
    .map((value, index) => {
      const px = (index / (values.length - 1)) * width;
      const py = height - ((value - min) / span) * (height - 4) - 2;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss over gradient steps">
      <polyline points={points} />
    </svg>
  );
}
