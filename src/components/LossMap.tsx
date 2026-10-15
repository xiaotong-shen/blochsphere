import { useEffect, useRef } from "react";
import type { LossGrid } from "../quantum/qnn";
import { wrapPi } from "../view/coords";

export type StepMark = { theta: number; phi: number };

type Props = {
  grid: LossGrid;
  current: StepMark;
  history: StepMark[];
};

function colorAt(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const r = Math.round(28 + clamped * 196);
  const g = Math.round(78 + (1 - clamped) * 48 + clamped * 70);
  const b = Math.round(112 + (1 - clamped) * 70 - clamped * 40);
  return [r, g, b];
}

export function LossMap({ grid, current, history }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const draw = () => {
      const width = parent.clientWidth;
      const height = parent.clientHeight;
      if (width < 2 || height < 2) return;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

      const padLeft = 36;
      const padRight = 12;
      const padTop = 10;
      const padBottom = 28;
      const plotW = width - padLeft - padRight;
      const plotH = height - padTop - padBottom;
      const span = grid.max - grid.min || 1;
      const cellW = plotW / grid.n;
      const cellH = plotH / grid.n;

      ctx.clearRect(0, 0, width, height);
      for (let row = 0; row < grid.n; row += 1) {
        for (let col = 0; col < grid.n; col += 1) {
          const t = (grid.values[row * grid.n + col] - grid.min) / span;
          const [r, g, b] = colorAt(t);
          ctx.fillStyle = `rgb(${r} ${g} ${b})`;
          ctx.fillRect(padLeft + col * cellW, padTop + row * cellH, cellW + 0.6, cellH + 0.6);
        }
      }

      const toPixel = (theta: number, phi: number) => {
        const wrappedTheta = wrapPi(theta);
        const wrappedPhi = wrapPi(phi);
        return [
          padLeft + ((wrappedTheta + Math.PI) / (2 * Math.PI)) * plotW,
          padTop + ((Math.PI - wrappedPhi) / (2 * Math.PI)) * plotH,
        ];
      };

      if (history.length > 1) {
        ctx.beginPath();
        history.forEach((step, index) => {
          const [x, y] = toPixel(step.theta, step.phi);
          if (index === 0) {
            ctx.moveTo(x, y);
            return;
          }
          const [previousX, previousY] = toPixel(history[index - 1].theta, history[index - 1].phi);
          if (Math.hypot(x - previousX, y - previousY) > Math.min(plotW, plotH) * 0.45) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = "#f3efe4";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      const [cx, cy] = toPixel(current.theta, current.phi);
      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#101218";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#f3efe4";
      ctx.stroke();

      ctx.fillStyle = "#a39c90";
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("−π", padLeft, height - 8);
      ctx.fillText("θ", padLeft + plotW / 2 - 4, height - 8);
      ctx.fillText("π", padLeft + plotW - 8, height - 8);
      ctx.save();
      ctx.translate(12, padTop + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("φ", 0, 0);
      ctx.restore();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [current.phi, current.theta, grid, history]);

  return <canvas ref={canvasRef} className="loss-canvas" aria-label="Loss over theta and phi" />;
}
