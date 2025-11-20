// frontend/src/App.tsx
import React, { useEffect, useRef, useState } from "react";

/**
 * App.tsx
 * Mode: Camera + Upload (Option C: AI-style Dropzone)
 * - Works with @mediapipe/tasks-vision (Tasks API) if present
 * - Graceful fallback if model fails to load
 *
 * Replace your existing App.tsx with this file.
 */

type Landmark = { x: number; y: number; z?: number };

export default function App(): JSX.Element {
  // DOM refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Tasks model refs
  const handLandmarkerRef = useRef<any | null>(null);
  const modelLoadedRef = useRef<boolean>(false);

  // Loop & camera control
  const rafIdRef = useRef<number | null>(null);
  const lastFrameTime = useRef<number>(performance.now());

  // UI and app state
  const [mode, setMode] = useState<"camera" | "upload">("camera");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null); // from camera capture
  const [uploadedImage, setUploadedImage] = useState<string | null>(null); // from upload
  const [prediction, setPrediction] = useState<string>("");
  const [fps, setFps] = useState<number>(0);
  const [statusMsg, setStatusMsg] = useState<string>("Loading model...");

  // UI sizing constants (prevents shrinking)
  const CARD_W = 600;
  const CARD_H = 440;

  // -------------------------
  // Load Tasks Model (safe)
  // -------------------------
  useEffect(() => {
    let canceled = false;

    async function loadModelSafe() {
      try {
        // dynamic import to avoid breaking bundler if package absent
        const mod = await import("@mediapipe/tasks-vision");
        const { HandLandmarker, FilesetResolver } = mod as any;

        // try to use local public paths first (ensure frontend/public/wasm and /models exist)
        const visionFileset = await FilesetResolver.forVisionTasks("/wasm");

        const handLandmarker = await HandLandmarker.createFromOptions(
          visionFileset,
          {
            baseOptions: {
              modelAssetPath: "/models/hand_landmarker.task", // put .task in public/models/
            },
            runningMode: "VIDEO",
            numHands: 1,
          }
        );

        if (canceled) {
          handLandmarker?.close?.();
          return;
        }

        handLandmarkerRef.current = handLandmarker;
        modelLoadedRef.current = true;
        setStatusMsg("Model loaded.");
      } catch (err) {
        console.warn("Model load failed — landmarking disabled.", err);
        modelLoadedRef.current = false;
        setStatusMsg(
          "Model not loaded. Landmarks disabled (camera & upload still work)."
        );
      }
    }

    loadModelSafe();

    return () => {
      canceled = true;
      try {
        if (handLandmarkerRef.current?.close) handLandmarkerRef.current.close();
      } catch {}
    };
  }, []);

  // -------------------------
  // Camera helpers
  // -------------------------
  const stopCameraAndLoop = () => {
    // cancel RAF
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    // stop media tracks
    const v = videoRef.current;
    if (v && v.srcObject) {
      const s = v.srcObject as MediaStream;
      s.getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }

    setIsCameraOn(false);
  };

  useEffect(() => {
    // on unmount cleanup
    return () => {
      stopCameraAndLoop();
      try {
        if (handLandmarkerRef.current?.close)
          handLandmarkerRef.current.close();
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------
  // Start Camera
  // -------------------------
  const startCamera = async () => {
    setPrediction("");
    setCapturedImage(null);
    setUploadedImage(null);

    try {
      // request camera with ideal resolution matching card
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: CARD_W }, height: { ideal: CARD_H } },
        audio: false,
      });

      if (!videoRef.current) return;

      // stop previous if running
      stopCameraAndLoop();

      videoRef.current.srcObject = stream;
      // playsInline & muted help autoplay in browsers
      videoRef.current.playsInline = true;
      videoRef.current.muted = true;

      await videoRef.current.play().catch(() => {}); // ignore if autoplay blocked

      setIsCameraOn(true);
      lastFrameTime.current = performance.now();
      runDetectionLoop();
    } catch (err) {
      console.error("Camera start failed:", err);
      setStatusMsg("Camera access denied or not available.");
    }
  };

  // -------------------------
  // Detection Loop (camera)
  // -------------------------
  const runDetectionLoop = () => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    // ensure canvas logical pixels match video
    overlay.width = video.videoWidth || CARD_W;
    overlay.height = video.videoHeight || CARD_H;

    const step = async () => {
      if (!videoRef.current || !overlayRef.current) return;

      const now = performance.now();
      const delta = now - lastFrameTime.current || 16;
      lastFrameTime.current = now;
      setFps(Math.round(1000 / delta));

      // clear overlay
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      // only run detection if model loaded
      if (modelLoadedRef.current && handLandmarkerRef.current) {
        try {
          const res = await handLandmarkerRef.current.detectForVideo(
            videoRef.current,
            now
          );
          if (res && Array.isArray(res.landmarks) && res.landmarks.length > 0) {
            drawLandmarks(ctx, res.landmarks[0]);
          }
        } catch (err) {
          // degrade gracefully
          console.warn("Detection error:", err);
        }
      }

      rafIdRef.current = requestAnimationFrame(step);
    };

    if (!rafIdRef.current) rafIdRef.current = requestAnimationFrame(step);
  };

  // -------------------------
  // Draw landmarks (lines + points)
  // -------------------------
  function drawLandmarks(ctx: CanvasRenderingContext2D, landmarks: Landmark[]) {
    ctx.save();
    ctx.fillStyle = "rgba(255,0,255,0.95)";
    ctx.strokeStyle = "rgba(0,255,255,0.95)";
    ctx.lineWidth = 2;

    const bones: Array<[number, number]> = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [0, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [5, 9],
      [9, 10],
      [10, 11],
      [11, 12],
      [9, 13],
      [13, 14],
      [14, 15],
      [15, 16],
      [13, 17],
      [17, 18],
      [18, 19],
      [19, 20],
    ];

    ctx.beginPath();
    for (const [a, b] of bones) {
      if (!landmarks[a] || !landmarks[b]) continue;
      const ax = landmarks[a].x * ctx.canvas.width;
      const ay = landmarks[a].y * ctx.canvas.height;
      const bx = landmarks[b].x * ctx.canvas.width;
      const by = landmarks[b].y * ctx.canvas.height;
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();

    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * ctx.canvas.width, p.y * ctx.canvas.height, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  // -------------------------
  // Capture frame (from camera)
  // -------------------------
  const captureFrame = () => {
    if (!videoRef.current || !captureCanvasRef.current) return;

    // stop overlay loop to reduce CPU while previewing
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    const canvas = captureCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const S = 360; // thumbnail size
    canvas.width = S;
    canvas.height = S;

    const vid = videoRef.current;
    const vw = vid.videoWidth || CARD_W;
    const vh = vid.videoHeight || CARD_H;

    const side = Math.min(vw, vh);
    const sx = Math.max(0, (vw - side) / 2);
    const sy = Math.max(0, (vh - side) / 2);

    ctx.drawImage(vid, sx, sy, side, side, 0, 0, S, S);

    setCapturedImage(canvas.toDataURL("image/png"));

    try {
      vid.pause();
    } catch {}
  };

  // -------------------------
  // Upload handling (dropzone)
  // -------------------------
  const handleFilePicked = (file?: File | null) => {
    if (!file) return;
    // revoke previous
    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage);
    }
    const url = URL.createObjectURL(file);
    setUploadedImage(url);
    setPrediction("");
  };

  const onDropOrSelect = (ev: React.DragEvent | React.ChangeEvent<HTMLInputElement>) => {
    ev.preventDefault();
    if ("dataTransfer" in ev) {
      const dt = ev.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleFilePicked(dt.files[0]);
      }
    } else {
      const input = ev.target as HTMLInputElement;
      if (input.files && input.files[0]) handleFilePicked(input.files[0]);
    }
  };

  // -------------------------
  // Mode switching helpers
  // -------------------------
  const switchToMode = (m: "camera" | "upload") => {
    setMode(m);
    setPrediction("");
    // If switching to upload, stop camera so tracks don't run
    if (m === "upload") {
      stopCameraAndLoop();
    }
    // If switching to camera, clear uploaded image
    if (m === "camera") {
      if (uploadedImage) {
        URL.revokeObjectURL(uploadedImage);
        setUploadedImage(null);
      }
    }
  };

  // -------------------------
  // Retake behavior (works for both)
  // -------------------------
  const handleRetake = async () => {
    setPrediction("");
    setCapturedImage(null);
    // if in upload mode, clear uploaded preview
    if (mode === "upload") {
      if (uploadedImage) {
        URL.revokeObjectURL(uploadedImage);
        setUploadedImage(null);
      }
      // focus file input
      fileInputRef.current?.focus();
    } else {
      // resume camera loop
      if (videoRef.current) {
        try {
          await videoRef.current.play();
        } catch {}
      }
      if (!rafIdRef.current && isCameraOn) {
        lastFrameTime.current = performance.now();
        runDetectionLoop();
      } else if (!isCameraOn) {
        startCamera();
      }
    }
  };

  // -------------------------
  // Predict (from captured or uploaded)
  // -------------------------
  const predictBackend = async () => {
    const source = capturedImage ?? uploadedImage;
    if (!source) {
      setStatusMsg("Please capture or upload an image first.");
      return;
    }

    setStatusMsg("Sending to backend...");
    try {
      const blob = await (await fetch(source)).blob();
      const fd = new FormData();
      fd.append("file", blob, "input.png");

      const res = await fetch("https://hand-gesture-recognition-nbhx.onrender.com/predict/", 
        {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();
      setPrediction(String(data.prediction ?? "unknown"));
      setStatusMsg("Prediction received.");
    } catch (err) {
      console.error("Predict error:", err);
      setStatusMsg("Prediction failed. Check backend.");
    }
  };

  // -------------------------
  // Prevent default for drag events on window
  // -------------------------
  useEffect(() => {
    const handler = (e: DragEvent) => e.preventDefault();
    window.addEventListener("dragover", handler);
    window.addEventListener("drop", handler);
    return () => {
      window.removeEventListener("dragover", handler);
      window.removeEventListener("drop", handler);
    };
  }, []);

  // -------------------------
  // UI Render
  // -------------------------
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-[#06101b] to-[#071433] p-6">
      <div
        className="bg-white/5 backdrop-blur-md border border-white/6 rounded-3xl shadow-2xl p-6"
        style={{ width: CARD_W + 40 }}
      >
        {/* Header + Tabs */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-semibold text-white">Hand Gesture AI</h1>
            <p className="text-sm text-gray-300 mt-1">
              Premium demo • Camera & Upload (AI Dropzone)
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => switchToMode("camera")}
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                mode === "camera"
                  ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white"
                  : "bg-white/3 text-gray-200"
              }`}
            >
              Camera
            </button>
            <button
              onClick={() => switchToMode("upload")}
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                mode === "upload"
                  ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white"
                  : "bg-white/3 text-gray-200"
              }`}
            >
              Upload
            </button>
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm text-gray-300">{statusMsg}</div>
          <div className="text-sm text-gray-300">
            FPS: <span className="text-green-400 font-medium">{fps}</span>
          </div>
        </div>

        {/* Camera/Upload Card */}
        <div
          className="mx-auto rounded-2xl overflow-hidden bg-black/80 border border-white/8"
          style={{ width: CARD_W, height: CARD_H }}
        >
          {/* CAMERA MODE */}
          {mode === "camera" && (
            <>
              {!capturedImage ? (
                <div style={{ position: "relative", width: "100%", height: "100%" }}>
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    playsInline
                    muted
                  />
                  <canvas
                    ref={overlayRef}
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      width: "100%",
                      height: "100%",
                      pointerEvents: "none",
                    }}
                  />
                </div>
              ) : (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: "100%",
                    height: "100%",
                    background:
                      "linear-gradient(180deg, rgba(8,10,15,0.6), rgba(12,14,20,0.8))",
                  }}
                >
                  <img
                    src={capturedImage}
                    alt="captured"
                    style={{
                      width: 340,
                      height: 340,
                      objectFit: "cover",
                      borderRadius: 18,
                      border: "2px solid rgba(255,255,255,0.04)",
                    }}
                  />
                </div>
              )}
            </>
          )}

          {/* UPLOAD MODE */}
          {mode === "upload" && (
            <div
              className="w-full h-full flex items-center justify-center"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => onDropOrSelect(e)}
            >
              {!uploadedImage ? (
                <div
                  className="w-[80%] h-[70%] flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/10 bg-gradient-to-b from-black/40 to-black/20 text-center p-6 cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <svg
                    width="48"
                    height="48"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="mb-4 text-white/80"
                  >
                    <path d="M12 2v14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M5 13l7-7 7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M21 21H3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>

                  <div className="text-white font-semibold mb-2">Drop an image or click to upload</div>
                  <div className="text-sm text-gray-400">PNG, JPG — will be sent to the same prediction endpoint</div>
                  <div className="mt-4">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        fileInputRef.current?.click();
                      }}
                      className="px-4 py-2 mt-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow"
                    >
                      Choose file
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "100%", height: "100%" }}>
                  <img
                    src={uploadedImage}
                    alt="uploaded"
                    style={{ width: 340, height: 340, objectFit: "cover", borderRadius: 18 }}
                  />
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onDropOrSelect}
              />
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="mt-5 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                // Camera mode button either starts or captures
                if (mode === "camera") {
                  if (!isCameraOn) startCamera();
                  else captureFrame();
                } else {
                  // upload mode: open file chooser
                  fileInputRef.current?.click();
                }
              }}
              className="px-5 py-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold shadow transform hover:scale-105 transition"
            >
              {mode === "camera"
                ? !isCameraOn
                  ? "🎥 Start Camera"
                  : "📸 Capture"
                : "📁 Upload Image"}
            </button>

            <button
              onClick={() => {
                // Stop or Retake
                if (mode === "camera") {
                  if (capturedImage) handleRetake();
                  else stopCameraAndLoop();
                } else {
                  // upload mode
                  if (uploadedImage) {
                    // clear uploaded preview
                    if (uploadedImage) URL.revokeObjectURL(uploadedImage);
                    setUploadedImage(null);
                    setPrediction("");
                  } else {
                    fileInputRef.current?.click();
                  }
                }
              }}
              className="px-4 py-2 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700 transition"
            >
              {mode === "camera" ? (capturedImage ? "🔄 Retake" : "⏹ Stop") : (uploadedImage ? "🔄 Clear" : "📂 Choose")}
            </button>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-sm text-gray-300 mr-3">Preview</div>
            <button
              onClick={predictBackend}
              className="px-4 py-2 rounded-full bg-white text-black font-semibold hover:opacity-90 transition"
            >
              🤖 Predict
            </button>
          </div>
        </div>

        {/* Prediction footer */}
        <div className="mt-4">
          {prediction ? (
            <div className="p-3 rounded-lg bg-gradient-to-r from-blue-800 to-purple-900 text-white font-medium">
              Prediction: <span className="ml-2 text-yellow-300">{prediction}</span>
            </div>
          ) : (
            <div className="text-sm text-gray-400">
              No prediction yet — capture or upload an image and press Predict.
            </div>
          )}
        </div>

        {/* hidden capture canvas */}
        <canvas ref={captureCanvasRef} style={{ display: "none" }} />
      </div>
    </div>
  );
}
