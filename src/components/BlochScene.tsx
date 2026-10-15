import { Line, OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import type { Bloch } from "../quantum/qubit";
import type { TrailSegment } from "../quantum/qnn";
import { toThree } from "../view/coords";

export type SphereDraw = {
  id: string;
  offset: [number, number, number];
  segments: TrailSegment[];
  tip: Bloch;
  caption: string;
};

function Axis({
  dx,
  dy,
  dz,
  color,
  label,
  lx,
  ly,
  lz,
}: {
  dx: number;
  dy: number;
  dz: number;
  color: string;
  label: string;
  lx: number;
  ly: number;
  lz: number;
}) {
  const arrow = useMemo(
    () => new THREE.ArrowHelper(new THREE.Vector3(dx, dy, dz), new THREE.Vector3(0, 0, 0), 1.28, color, 0.08, 0.045),
    [color, dx, dy, dz],
  );
  return (
    <group>
      <primitive object={arrow} />
      <Label position={[lx, ly, lz]} text={label} color={color} />
    </group>
  );
}

function Label({
  position,
  text,
  color,
}: {
  position: [number, number, number];
  text: string;
  color: string;
}) {
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (!context) return null;
    const font = "600 64px 'Avenir Next', sans-serif";
    context.font = font;
    const textWidth = Math.ceil(context.measureText(text).width);
    canvas.width = textWidth + 32;
    canvas.height = 96;
    context.font = font;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = color;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, canvas.width / 2, canvas.height / 2);
    const map = new THREE.CanvasTexture(canvas);
    map.needsUpdate = true;
    return { map, aspect: canvas.width / canvas.height };
  }, [color, text]);
  if (!texture) return null;
  return (
    <sprite position={position} scale={[0.2 * texture.aspect, 0.2, 1]}>
      <spriteMaterial map={texture.map} transparent depthWrite={false} />
    </sprite>
  );
}

function StateArrow({ tip, color }: { tip: Bloch; color: string }) {
  const [x, y, z] = toThree(tip);
  const length = Math.hypot(x, y, z);
  const arrow = useMemo(
    () => new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 1, color, 0.13, 0.06),
    [color],
  );
  useLayoutEffect(() => {
    if (length < 1e-3) {
      arrow.visible = false;
      return;
    }
    arrow.visible = true;
    arrow.setDirection(new THREE.Vector3(x / length, y / length, z / length));
    arrow.setLength(length, Math.min(0.16, length * 0.38), Math.min(0.08, length * 0.2));
  }, [arrow, length, x, y, z]);

  return (
    <group>
      <primitive object={arrow} />
      <mesh position={[x, y, z]}>
        <sphereGeometry args={[0.045, 18, 18]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35} />
      </mesh>
    </group>
  );
}

function trailPoint(point: Bloch): [number, number, number] {
  const [x, y, z] = toThree(point);
  const length = Math.hypot(x, y, z);
  if (length > 0.985) {
    const scale = 1.02 / length;
    return [x * scale, y * scale, z * scale];
  }
  return [x, y, z];
}

function TrailLine({ points, color }: { points: Bloch[]; color: string }) {
  const positions = points.map(trailPoint);
  return <Line points={positions} color={color} lineWidth={2.5} />;
}

function Trail({ segments }: { segments: TrailSegment[] }) {
  return (
    <>
      {segments.map((segment) =>
        segment.points.length < 2 ? null : (
          <TrailLine key={segment.name} points={segment.points} color={segment.color} />
        ),
      )}
    </>
  );
}

function Pole({ position, text }: { position: [number, number, number]; text: string }) {
  return <Label position={position} text={text} color="#ebe6dc" />;
}

function OneSphere({ draw, showLabels }: { draw: SphereDraw; showLabels: boolean }) {
  return (
    <group position={draw.offset}>
      <mesh>
        <sphereGeometry args={[1, 48, 32]} />
        <meshBasicMaterial color="#9aabbd" wireframe transparent opacity={0.28} depthWrite={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.99, 48, 32]} />
        <meshStandardMaterial color="#243244" transparent opacity={0.12} roughness={0.35} depthWrite={false} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1, 0.005, 8, 96]} />
        <meshBasicMaterial color="#d9d3c7" />
      </mesh>
      <Axis dx={1} dy={0} dz={0} color="#e07a5f" label="x" lx={1.42} ly={0} lz={0} />
      <Axis dx={0} dy={0} dz={-1} color="#8fbf9f" label="y" lx={0} ly={0} lz={-1.42} />
      <Axis dx={0} dy={1} dz={0} color="#7eb6ff" label="z" lx={0} ly={1.42} lz={0} />
      {showLabels ? (
        <>
          <Pole position={[0, 1.2, 0]} text="|0⟩" />
          <Pole position={[0, -1.22, 0]} text="|1⟩" />
          <Pole position={[1.22, 0, 0]} text="|+⟩" />
          <Pole position={[-1.22, 0, 0]} text="|−⟩" />
        </>
      ) : null}
      <Trail segments={draw.segments} />
      <StateArrow tip={draw.tip} color="#f3efe4" />
      <Label position={[0, -1.55, 0]} text={draw.caption} color="#a39c90" />
    </group>
  );
}

function CameraRig({ two }: { two: boolean }) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls) as { target: THREE.Vector3; update: () => void } | null;
  useLayoutEffect(() => {
    const next = two ? ([0.3, 2.05, 6.5] as const) : ([2.55, 1.62, 2.85] as const);
    camera.position.set(next[0], next[1], next[2]);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = two ? 38 : 36;
      camera.updateProjectionMatrix();
    }
    camera.lookAt(0, 0, 0);
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [camera, controls, two]);
  return null;
}

export function BlochScene({ spheres }: { spheres: SphereDraw[] }) {
  const two = spheres.length > 1;
  return (
    <Canvas
      camera={{ position: [2.55, 1.62, 2.85], fov: 36 }}
      gl={{ antialias: true }}
      onCreated={({ gl }) => gl.setClearColor("#101218")}
    >
      <CameraRig two={two} />
      <ambientLight intensity={0.85} />
      <directionalLight position={[4, 5, 3]} intensity={1.05} />
      {spheres.map((sphere) => (
        <OneSphere key={sphere.id} draw={sphere} showLabels={!two} />
      ))}
      <OrbitControls makeDefault enablePan={false} />
    </Canvas>
  );
}
