import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Camera, Play, Square } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";
import { PredictionPanel } from "@/components/signease/PredictionPanel";

const SEND_INTERVAL_MS = 500;
const BACKEND_URL = "http://localhost:8001/predict";
const CONFIDENCE_THRESHOLD_PERCENT = 30;
const BUFFER_SIZE = 5;
const MAJORITY_COUNT = 2;

const SignToText = () => {
  const navigate = useNavigate();
  const webcamRef = useRef<Webcam>(null);
  const [isCommunicating, setIsCommunicating] = useState(false);
  const [detectedText, setDetectedText] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [statusText, setStatusText] = useState<string>("");
  const [isStabilizing, setIsStabilizing] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [rawConfidence, setRawConfidence] = useState<number | null>(null);
  const [rawPrediction, setRawPrediction] = useState<string>("");
  const intervalRef = useRef<number | null>(null);
  const predBufferRef = useRef<Array<{ text: string; conf: number }>>([]);

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user" as const,
  };

  const sendFrameOnce = async () => {
    const screenshot = webcamRef.current?.getScreenshot();
    if (!screenshot) return;

    try {
      if (!statusText && !detectedText) setStatusText("Connecting to AI...");

      console.log("Sending frame to backend...");
      const resp = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: screenshot }),
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      const data = await resp.json();
      console.log("Response received:", data);
      console.log("RAW BACKEND RESPONSE:", data);
      const text = String(data.text ?? "").trim();
      const confPercent = Math.round(Number(data.confidence ?? 0) * 100);
      setRawConfidence(Number.isFinite(confPercent) ? confPercent : null);
      setRawPrediction(text);

      if (!text) {
        // Fallback so examiner can see state updates even if backend returns empty text.
        if (confPercent > 0) setRawPrediction("Unknown");
        setStatusText("Analyzing...");
        setIsStabilizing(true);
        return;
      }

      if (confPercent < CONFIDENCE_THRESHOLD_PERCENT) {
        setStatusText("Keep hand steady");
        setIsStabilizing(true);
        return;
      }

      // Add to rolling buffer (last 5)
      const buf = predBufferRef.current;
      buf.push({ text, conf: confPercent });
      while (buf.length > BUFFER_SIZE) buf.shift();

      // Majority vote
      const counts = new Map<string, number>();
      for (const p of buf) counts.set(p.text, (counts.get(p.text) ?? 0) + 1);
      let bestText = "";
      let bestCount = 0;
      for (const [t, c] of counts.entries()) {
        if (c > bestCount) {
          bestText = t;
          bestCount = c;
        }
      }

      if (bestCount >= MAJORITY_COUNT) {
        const matching = buf.filter((p) => p.text === bestText);
        const avgConf =
          matching.reduce((sum, p) => sum + p.conf, 0) / Math.max(1, matching.length);

        setDetectedText(bestText);
        setConfidence(Math.round(avgConf));
        setStatusText("");
        setIsStabilizing(false);
      } else {
        setStatusText("Analyzing...");
        setIsStabilizing(true);
      }
    } catch (e) {
      console.error("Backend connection/predict failed:", e);
      setStatusText("Connecting to AI...");
      setIsStabilizing(false);
    }
  };

  const startCommunicating = () => {
    if (intervalRef.current !== null) return; // avoid duplicate intervals
    setIsCommunicating(true);
    setStatusText("Connecting to AI...");
    setDetectedText("");
    setConfidence(0);
    setIsStabilizing(false);
    setRawConfidence(null);
    setRawPrediction("");
    predBufferRef.current = [];

    // send immediately, then every 500ms
    void sendFrameOnce();
    intervalRef.current = window.setInterval(() => {
      void sendFrameOnce();
    }, SEND_INTERVAL_MS);
  };

  const stopCommunicating = () => {
    setIsCommunicating(false);
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setStatusText("");
    setIsStabilizing(false);
    predBufferRef.current = [];
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-background px-4 py-6 md:px-8">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="max-w-6xl mx-auto w-full"
      >
        <div className="flex items-center gap-4 mb-8">
          <Button variant="ghost" onClick={() => navigate("/")} className="text-white">
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Button>
          <h1 className="text-3xl font-bold text-white">Sign Language to Text</h1>
        </div>

        <div className="grid gap-6">
          <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr] items-stretch">
          <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6 h-full">
            <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
              <p className="text-sm text-blue-200">
                🎯 <strong>ASL Recognition:</strong> Optimized for a 10-word assistive vocabulary: Food, Hello, Help, More, No, Please, Sad, Thank You, Water, and Yes. Powered by Hybrid VideoMAE for 3D motion tracking.
              </p>
            </div>

            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Camera className="w-5 h-5" />
              Camera Input
            </h2>
            
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-4 min-h-[280px]">
              <Webcam
                ref={webcamRef}
                className="w-full h-full object-cover"
                mirrored={true}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                onUserMedia={() => {
                  console.log("Webcam ready");
                  setIsCameraReady(true);
                }}
                onUserMediaError={(err) => {
                  console.log("Webcam permission/device error:", err);
                  setIsCameraReady(false);
                }}
              />

              {!isCameraReady && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="px-4 py-2 rounded-lg bg-black/70 border border-white/10 text-white text-lg font-semibold">
                    Loading Camera...
                  </div>
                </div>
              )}

              {(statusText || detectedText) && (
                <div className="absolute top-4 left-4 right-4 flex justify-center pointer-events-none">
                  <div className="px-4 py-2 rounded-full bg-black/70 border border-white/10 text-white text-lg font-semibold">
                    {statusText ? statusText : `${detectedText}  •  ${confidence}%`}
                  </div>
                </div>
              )}

              {/* Hand placement guide overlay - dotted square */}
              <div className="absolute inset-0 pointer-events-none">
                {/* Dotted border square in center */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 border-2 border-dashed border-white/60 rounded-lg">
                  {/* Corner markers for better visibility */}
                  <div className="absolute -top-1 -left-1 w-3 h-3 bg-white/80 rounded-full"></div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-white/80 rounded-full"></div>
                  <div className="absolute -bottom-1 -left-1 w-3 h-3 bg-white/80 rounded-full"></div>
                  <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-white/80 rounded-full"></div>
                </div>

                {/* Hand icon guide */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center">
                  <div className="text-white/40 text-6xl opacity-30">
                    ✋
                  </div>
                </div>

                {/* Instruction text */}
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 text-center">
                  <p className="text-white/70 text-sm bg-black/50 px-3 py-1 rounded-full">
                    Position hand in center square for best recognition
                  </p>
                </div>
              </div>
            </div>

            <div className="flex justify-center gap-4">
              <Button
                variant={isCommunicating ? "destructive" : "hero"}
                onClick={isCommunicating ? stopCommunicating : startCommunicating}
                className="min-w-[140px]"
              >
                {isCommunicating ? (
                  <>
                    <Square className="w-4 h-4" />
                    Stop
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Start Communicating
                  </>
                )}
              </Button>
            </div>
          </Card>

          <PredictionPanel
            isCommunicating={isCommunicating}
            statusText={statusText}
            prediction={detectedText}
            confidencePercent={confidence}
            stabilizing={isStabilizing}
            debugConfidencePercent={
              statusText === "Keep hand steady" || statusText === "Analyzing..."
                ? rawConfidence
                : null
            }
            debugRawPrediction={
              statusText === "Keep hand steady" || statusText === "Analyzing..."
                ? (rawPrediction || (rawConfidence && rawConfidence > 0 ? "Unknown" : ""))
                : null
            }
          />
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default SignToText;