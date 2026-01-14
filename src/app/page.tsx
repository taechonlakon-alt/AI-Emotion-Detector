"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

// Helper เพื่อเลือกสีและ Emoji ตามอารมณ์ (ปรับตาม Class ที่เทรนมาจริงได้เลย)
const getEmotionStyle = (emotion: string) => {
  const e = emotion.toLowerCase();
  if (e.includes("happy") || e.includes("joy")) return { color: "text-green-400", emoji: "😄", label: "Happy" };
  if (e.includes("sad")) return { color: "text-blue-400", emoji: "😢", label: "Sad" };
  if (e.includes("angry")) return { color: "text-red-500", emoji: "😡", label: "Angry" };
  if (e.includes("surprise")) return { color: "text-yellow-400", emoji: "😲", label: "Surprise" };
  if (e.includes("neutral")) return { color: "text-gray-300", emoji: "😐", label: "Neutral" };
  return { color: "text-white", emoji: "🤔", label: emotion }; // Default
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("Initializing...");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // --- Logic เดิม (Load OpenCV) ---
  async function loadOpenCV() {
    if (typeof window === "undefined") return;

    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;

      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));

        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };

        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };

      script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  // --- Logic เดิม (Load Cascade) ---
  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");
    const data = new Uint8Array(await res.arrayBuffer());

    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_unlink(cascadePath);
    } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);

    const faceCascade = new cv.CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  // --- Logic เดิม (Load Model) ---
  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;

    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("โหลด classes.json ไม่สำเร็จ");
    classesRef.current = await clsRes.json();
  }

  // --- Logic เดิม (Start Camera) ปรับปรุง State เล็กน้อย ---
  async function startCamera() {
    setStatus("Requesting camera access...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      
      setStatus("Active");
      setIsCameraActive(true);
      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`Camera Error: ${e.message}`);
    }
  }

  // --- Logic เดิม (Preprocess) ---
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = imgData[i * 4 + 0] / 255;
        const g = imgData[i * 4 + 1] / 255;
        const b = imgData[i * 4 + 2] / 255;
        float[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  // --- Logic เดิม (Softmax) ---
  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // --- Logic เดิม (Loop) ---
  async function loop() {
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        requestAnimationFrame(loop);
        return;
      }

      if (video.paused || video.ended) return;

      const ctx = canvas.getContext("2d")!;
      if(canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      const msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

      let bestRect: any = null;
      let bestArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) {
          bestArea = area;
          bestRect = r;
        }
        
        // วาดกรอบหน้า (ปรับสีให้เข้ากับธีม)
        ctx.strokeStyle = "#00ffcc"; // Cyan
        ctx.lineWidth = 3;
        
        // วาดมุมโค้งนิดหน่อย (Optional: ถ้าจะวาดแค่ rect ธรรมดาให้ใช้ strokeRect)
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width;
        faceCanvas.height = bestRect.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(canvas, bestRect.x, bestRect.y, bestRect.width, bestRect.height, 0, 0, bestRect.width, bestRect.height);

        const input = preprocessToTensor(faceCanvas);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const outName = session.outputNames[0];
        const logits = out[outName].data as Float32Array;
        const probs = softmax(logits);
        
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        const detectedEmotion = classes[maxIdx] ?? `class_${maxIdx}`;
        setEmotion(detectedEmotion);
        setConf(probs[maxIdx] ?? 0);

        // วาด Label บน Canvas
        const text = `${detectedEmotion} ${(probs[maxIdx] * 100).toFixed(0)}%`;
        ctx.font = "bold 20px Prompt, sans-serif";
        const textWidth = ctx.measureText(text).width;
        
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(bestRect.x, bestRect.y - 30, textWidth + 10, 30);
        
        ctx.fillStyle = "#00ffcc";
        ctx.fillText(text, bestRect.x + 5, bestRect.y - 8);
      } else {
        // Reset ถ้าไม่เจอหน้า
        // setEmotion("-");
        // setConf(0);
      }

      src.delete();
      gray.delete();
      faces.delete();

      requestAnimationFrame(loop);
    } catch (e: any) {
      console.error(e);
      setStatus(`Error: ${e?.message ?? e}`);
    }
  }

  // --- Boot Sequence ---
  useEffect(() => {
    (async () => {
      try {
        setIsLoading(true);
        setStatus("Loading OpenCV...");
        await loadOpenCV();

        setStatus("Loading Haar Cascade...");
        await loadCascade();

        setStatus("Loading AI Model...");
        await loadModel();

        setStatus("Ready");
        setIsLoading(false);
      } catch (e: any) {
        setStatus(`Setup Failed: ${e?.message ?? e}`);
        setIsLoading(false);
      }
    })();
  }, []);

  const emotionStyle = getEmotionStyle(emotion);

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col items-center justify-center p-4 font-sans selection:bg-purple-500 selection:text-white">
      
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 opacity-20 pointer-events-none">
        <div className="absolute top-[10%] left-[20%] w-72 h-72 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob"></div>
        <div className="absolute top-[10%] right-[20%] w-72 h-72 bg-cyan-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-[20%] left-[50%] w-72 h-72 bg-pink-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000 transform -translate-x-1/2"></div>
      </div>

      <div className="w-full max-w-4xl space-y-6">
        
        {/* Header */}
        <header className="text-center space-y-2">
          <h1 className="text-4xl md:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-500 tracking-tight">
            AI Emotion Detector
          </h1>
          <p className="text-neutral-400 text-sm md:text-base">
            Real-time Facial Expression Recognition powered by YOLO11 & OpenCV
          </p>
        </header>

        {/* Status Bar */}
        <div className="flex justify-center items-center gap-2 text-xs md:text-sm font-mono text-neutral-500 bg-neutral-900/50 py-1 px-3 rounded-full border border-neutral-800 w-fit mx-auto">
            <span className={`w-2 h-2 rounded-full ${isLoading ? 'bg-yellow-500 animate-pulse' : isCameraActive ? 'bg-green-500' : 'bg-red-500'}`}></span>
            {status}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Camera Viewport */}
          <div className="lg:col-span-2 relative aspect-video bg-black rounded-2xl overflow-hidden shadow-2xl border border-neutral-800 group">
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas
              ref={canvasRef}
              className={`w-full h-full object-cover transition-opacity duration-700 ${isCameraActive ? "opacity-100" : "opacity-30"}`}
            />
            
            {/* Camera Overlay when inactive */}
            {!isCameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 bg-neutral-900/40 backdrop-blur-sm z-10">
                {isLoading ? (
                    <div className="flex flex-col items-center gap-3">
                        <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="text-cyan-400 font-medium">Initializing AI Engine...</p>
                    </div>
                ) : (
                    <button
                    onClick={startCamera}
                    className="group relative inline-flex items-center justify-center px-8 py-3 font-bold text-white transition-all duration-200 bg-cyan-600 font-lg rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-600 hover:bg-cyan-500 hover:scale-105 active:scale-95 shadow-lg shadow-cyan-500/20"
                    >
                    <span className="mr-2">📸</span> Start Analysis
                    </button>
                )}
              </div>
            )}
            
            {/* Scanline Effect */}
            {isCameraActive && (
                <div className="absolute inset-0 pointer-events-none opacity-20 bg-[linear-gradient(transparent_0%,rgba(0,255,204,0.1)_50%,transparent_100%)] bg-[length:100%_4px] animate-scanline"></div>
            )}
          </div>

          {/* Stats Panel */}
          <div className="lg:col-span-1 flex flex-col gap-4">
            
            {/* Emotion Card */}
            <div className="flex-1 bg-neutral-900/60 backdrop-blur-md border border-neutral-800 rounded-2xl p-6 flex flex-col items-center justify-center text-center shadow-lg transition-colors duration-300">
                <h3 className="text-neutral-400 text-sm uppercase tracking-wider mb-2">Detected Emotion</h3>
                <div className={`text-6xl mb-4 transition-transform duration-300 ${isCameraActive && emotion !== "-" ? "scale-110" : "scale-100 opacity-50"}`}>
                    {emotionStyle.emoji}
                </div>
                <div className={`text-3xl font-bold ${emotionStyle.color} capitalize`}>
                    {emotion !== "-" ? emotionStyle.label : "Waiting..."}
                </div>
            </div>

            {/* Confidence Card */}
            <div className="bg-neutral-900/60 backdrop-blur-md border border-neutral-800 rounded-2xl p-6 shadow-lg">
                <div className="flex justify-between items-end mb-2">
                    <h3 className="text-neutral-400 text-sm uppercase tracking-wider">Confidence</h3>
                    <span className="text-2xl font-mono font-bold text-white">{(conf * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-4 bg-neutral-800 rounded-full overflow-hidden">
                    <div 
                        className={`h-full transition-all duration-300 ease-out ${conf > 0.7 ? 'bg-green-500' : conf > 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${conf * 100}%` }}
                    />
                </div>
            </div>

            {/* Hint */}
            <div className="p-4 rounded-xl bg-neutral-800/30 border border-neutral-700/50 text-xs text-neutral-500 text-center">
                Face well-lit and center for best accuracy.
            </div>

          </div>
        </div>

      </div>
    </main>
  );
}