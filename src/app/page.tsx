"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

// Serve WASM files from our own public folder (copied from node_modules at build time)
ort.env.wasm.wasmPaths = "/";

const CLASS_NAMES = ["Early-Turning", "Green", "Late-Turning", "Red", "Turning", "White"];

const RIPENESS_STYLE: Record<string, { color: string; emoji: string; stroke: string }> = {
  "Early-Turning": { color: "#fb923c", emoji: "🍓", stroke: "#fb923c" }, // Orange-ish
  "Green": { color: "#4ade80", emoji: "🟢", stroke: "#4ade80" },
  "Late-Turning": { color: "#f97316", emoji: "🍊", stroke: "#f97316" }, // Dark Orange
  "Red": { color: "#f87171", emoji: "🔴", stroke: "#f87171" }, // Red
  "Turning": { color: "#facc15", emoji: "🟡", stroke: "#facc15" }, // Yellow
  "White": { color: "#f1f5f9", emoji: "⚪", stroke: "#f1f5f9" }, // White/Grey
};

const getStyle = (cls: string) =>
  RIPENESS_STYLE[cls] ?? { color: "#fff", emoji: "🍓", stroke: "#fff" };

interface Detection {
  x1: number; y1: number; x2: number; y2: number;
  className: string;
  confidence: number;
}

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (aArea + bArea - inter + 1e-6);
}

function nms(boxes: Detection[], iouThresh = 0.45): Detection[] {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const result: Detection[] = [];
  while (sorted.length > 0) {
    const best = sorted.shift()!;
    result.push(best);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(best, sorted[i]) > iouThresh) sorted.splice(i, 1);
    }
  }
  return result;
}

const MODEL_SIZE = 640;
const CONF_THRESH = 0.25;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const animRef = useRef<number | null>(null);

  const [status, setStatus] = useState("Loading AI Model...");
  const [modelReady, setModelReady] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);

  // Load ONNX model on mount
  useEffect(() => {
    (async () => {
      try {
        setStatus("Downloading AI Model (~44 MB)...");
        const session = await ort.InferenceSession.create("/models/best.onnx", {
          executionProviders: ["wasm"],
        });
        sessionRef.current = session;
        setModelReady(true);
        setStatus("Ready — press Start");
      } catch (e: unknown) {
        setStatus(`Model Error: ${e instanceof Error ? e.message : String(e)}`);
      }
    })();
  }, []);

  async function startCamera() {
    try {
      let stream: MediaStream;
      try {
        // Try rear camera first
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
      } catch (err: any) {
        // Fallback to any camera if rear camera is not available
        console.warn("Rear camera not available, falling back to default camera", err);
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setIsCameraOn(true);
      setStatus("Active");
    } catch (e: unknown) {
      setStatus(`Camera Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function preprocessFrame(video: HTMLVideoElement): { data: Float32Array; scaleX: number; scaleY: number; padX: number; padY: number } {
    const offscreen = document.createElement("canvas");
    offscreen.width = MODEL_SIZE;
    offscreen.height = MODEL_SIZE;
    const ctx = offscreen.getContext("2d")!;

    // Letterbox
    const scale = Math.min(MODEL_SIZE / video.videoWidth, MODEL_SIZE / video.videoHeight);
    const sw = Math.round(video.videoWidth * scale);
    const sh = Math.round(video.videoHeight * scale);
    const padX = Math.floor((MODEL_SIZE - sw) / 2);
    const padY = Math.floor((MODEL_SIZE - sh) / 2);

    ctx.fillStyle = "#808080"; // YOLO expects grey padding
    ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    ctx.drawImage(video, padX, padY, sw, sh);

    const pixels = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    const float32 = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
    for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
      float32[i] = pixels[i * 4] / 255; // R
      float32[MODEL_SIZE * MODEL_SIZE + i] = pixels[i * 4 + 1] / 255; // G
      float32[2 * MODEL_SIZE * MODEL_SIZE + i] = pixels[i * 4 + 2] / 255; // B
    }
    return { data: float32, scaleX: scale, scaleY: scale, padX, padY };
  }

  const loopRef = useRef<(() => Promise<void>) | null>(null);

  // eslint-disable-next-line react-hooks/refs
  loopRef.current = async () => {

    function parseOutput(
      output: ort.Tensor,
      scaleX: number,
      scaleY: number,
      padX: number,
      padY: number,
    ): Detection[] {
      const data = output.data as Float32Array;
      const dims = output.dims; // [1, 4+nc, ...] or [1, ..., 4+nc]

      const isNHW = dims[1] < dims[2];
      const numBoxes = isNHW ? dims[2] : dims[1];
      const numClasses = (isNHW ? dims[1] : dims[2]) - 4;

      const boxes: Detection[] = [];
      for (let i = 0; i < numBoxes; i++) {
        let cx = 0, cy = 0, w = 0, h = 0, maxConf = 0, maxClass = 0;
        if (isNHW) {
          cx = data[0 * numBoxes + i];
          cy = data[1 * numBoxes + i];
          w = data[2 * numBoxes + i];
          h = data[3 * numBoxes + i];
          for (let c = 0; c < numClasses; c++) {
            const v = data[(4 + c) * numBoxes + i];
            if (v > maxConf) { maxConf = v; maxClass = c; }
          }
        } else {
          const base = i * (4 + numClasses);
          cx = data[base]; cy = data[base + 1]; w = data[base + 2]; h = data[base + 3];
          for (let c = 0; c < numClasses; c++) {
            const v = data[base + 4 + c];
            if (v > maxConf) { maxConf = v; maxClass = c; }
          }
        }
        if (maxConf < CONF_THRESH) continue;

        // Convert letterbox coords back to video coords
        const x1v = ((cx - w / 2) - padX) / scaleX;
        const y1v = ((cy - h / 2) - padY) / scaleY;
        const x2v = ((cx + w / 2) - padX) / scaleX;
        const y2v = ((cy + h / 2) - padY) / scaleY;

        if (x2v <= 0 || y2v <= 0) continue;

        boxes.push({
          x1: Math.max(0, x1v), y1: Math.max(0, y1v),
          x2: x2v, y2: y2v,
          className: CLASS_NAMES[maxClass] ?? `class${maxClass}`,
          confidence: maxConf,
        });
      }
      return nms(boxes);
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const session = sessionRef.current;

    if (!video || !canvas || !session || video.paused || video.ended || video.videoWidth === 0) {
      if (video?.srcObject) animRef.current = requestAnimationFrame(() => loopRef.current?.());
      return;
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const { data, scaleX, scaleY, padX, padY } = preprocessFrame(video);
      const tensor = new ort.Tensor("float32", data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
      const results = await session.run({ [session.inputNames[0]]: tensor });
      const output = results[session.outputNames[0]];
      const dets = parseOutput(output, scaleX, scaleY, padX, padY);
      setDetections(dets);

      const sx = canvas.width / video.videoWidth;
      const sy = canvas.height / video.videoHeight;

      dets.forEach((det) => {
        const style = getStyle(det.className);
        const x = det.x1 * sx;
        const y = det.y1 * sy;
        const bw = (det.x2 - det.x1) * sx;
        const bh = (det.y2 - det.y1) * sy;

        // Draw Bounding Box
        ctx.strokeStyle = style.stroke;
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.strokeRect(x, y, bw, bh);

        // Draw Label Background
        const label = `${style.emoji} ${det.className} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 16px sans-serif";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(x, y - 28, tw + 12, 28);

        // Draw Label Text
        ctx.fillStyle = style.color;
        ctx.fillText(label, x + 6, y - 8);
      });
    } catch (err) {
      console.error("Inference error:", err);
    }

    if (video.srcObject) animRef.current = requestAnimationFrame(() => loopRef.current?.());
  };

  useEffect(() => {
    if (isCameraOn) animRef.current = requestAnimationFrame(() => loopRef.current?.());
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isCameraOn]);

  // Count by class
  const countByClass = detections.reduce<Record<string, number>>((acc, d) => {
    acc[d.className] = (acc[d.className] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col items-center justify-center p-4 font-sans selection:bg-pink-500">
      {/* Background blobs */}
      <div className="absolute inset-0 -z-10 overflow-hidden opacity-20 pointer-events-none">
        <div className="absolute top-[10%] left-[20%] w-72 h-72 bg-red-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob" />
        <div className="absolute top-[10%] right-[20%] w-72 h-72 bg-green-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000" />
        <div className="absolute bottom-[20%] left-1/2 w-72 h-72 bg-yellow-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000 -translate-x-1/2" />
      </div>

      <div className="w-full max-w-4xl space-y-6 z-10">
        {/* Header */}
        <header className="text-center space-y-2">
          <h1 className="text-4xl md:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-red-400 to-green-500 tracking-tight">
            Strawberry Ripeness Checker
          </h1>
          <p className="text-neutral-400 text-sm">
            Real-time Detection · Powered by YOLO11 · Runs fully in your browser
          </p>
        </header>

        {/* Status badge */}
        <div className="flex justify-center">
          <div className="flex items-center gap-2 text-xs font-mono text-neutral-400 bg-neutral-900/60 py-1 px-4 rounded-full border border-neutral-800">
            <span className={`w-2 h-2 rounded-full ${status === "Active" ? "bg-green-500 animate-pulse" :
              status.includes("Ready") ? "bg-blue-400" :
                status.includes("Error") ? "bg-red-500" : "bg-yellow-400 animate-pulse"
              }`} />
            {status}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera canvas */}
          <div className="lg:col-span-2 relative aspect-[3/4] sm:aspect-video bg-black rounded-2xl overflow-hidden border border-neutral-800 shadow-2xl">
            <video ref={videoRef} className="hidden" playsInline muted autoPlay />
            <canvas ref={canvasRef} className="w-full h-full object-cover" />

            {!isCameraOn && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-neutral-900/70 backdrop-blur-sm z-10">
                {!modelReady ? (
                  <>
                    <div className="w-10 h-10 border-4 border-red-500 border-t-transparent rounded-full animate-spin" />
                    <p className="text-red-400 font-medium text-sm">{status}</p>
                    <p className="text-neutral-600 text-xs">Please wait, downloading ~44 MB…</p>
                  </>
                ) : (
                  <button
                    onClick={startCamera}
                    className="flex items-center gap-2 px-8 py-3 bg-red-600 hover:bg-red-500 rounded-full font-bold text-white transition-all hover:scale-105 active:scale-95 shadow-lg shadow-red-500/20"
                  >
                    <span>📸</span> Start Scanner
                  </button>
                )}
              </div>
            )}

            {/* Scanline */}
            {isCameraOn && (
              <div className="absolute inset-0 pointer-events-none opacity-10 bg-[linear-gradient(transparent_0%,rgba(255,0,0,0.15)_50%,transparent_100%)] bg-[length:100%_4px] animate-scanline" />
            )}
          </div>

          {/* Stats panel */}
          <div className="lg:col-span-1 flex flex-col gap-4">
            {/* Total count */}
            <div className="bg-neutral-900/60 backdrop-blur-md border border-neutral-800 rounded-2xl p-6 text-center">
              <h3 className="text-neutral-400 text-xs uppercase tracking-widest mb-2">Detected</h3>
              <div className="text-6xl font-extrabold text-white tabular-nums leading-none">
                {detections.length}
              </div>
              <div className="text-neutral-500 text-xs mt-2">strawberries in frame</div>
            </div>

            {/* Per-class breakdown */}
            <div className="bg-neutral-900/60 backdrop-blur-md border border-neutral-800 rounded-2xl p-5 flex flex-col gap-3 flex-1 overflow-y-auto max-h-[40vh] lg:max-h-full">
              <h3 className="text-neutral-400 text-xs uppercase tracking-widest sticky top-0 bg-neutral-900/90 py-1">Ripeness Breakdown</h3>
              {Object.keys(countByClass).length === 0 ? (
                <p className="text-neutral-600 text-xs text-center py-4 flex-1 flex items-center justify-center">
                  No strawberries detected
                </p>
              ) : (
                CLASS_NAMES.filter((cls) => countByClass[cls]).map((cls) => {
                  const s = getStyle(cls);
                  const count = countByClass[cls];
                  const pct = Math.round((count / detections.length) * 100);
                  return (
                    <div key={cls}>
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-base">{s.emoji}</span>
                          <span className="text-sm font-medium" style={{ color: s.color }}>{cls}</span>
                        </div>
                        <span className="text-white font-bold text-sm">{count}</span>
                      </div>
                      <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{ width: `${pct}%`, backgroundColor: s.color }}
                        />
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            <div className="p-4 rounded-xl bg-neutral-800/30 border border-neutral-700/50 text-xs text-neutral-500 text-center">
              Point your camera at strawberries to detect ripeness in real-time. Use rear camera if available.
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
