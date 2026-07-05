import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Search, RotateCcw, BadgeCheck, Copy, Languages } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

/** Default matches backend2 Flask (PORT=8001). Override with Vite: VITE_SIGNEASE_API_BASE */
export const SIGNease_API_BASE =
  (import.meta.env.VITE_SIGNEASE_API_BASE as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8001";

const API_BASE_CANDIDATES: readonly string[] = Array.from(
  new Set([SIGNease_API_BASE, "http://127.0.0.1:8001", "http://127.0.0.1:5000"]),
);

/** Display labels for the 10-word assistive vocabulary (folders use compact names, e.g. thankyou). */
export const SUPPORTED_VOCABULARY: readonly string[] = [
  "Food",
  "Hello",
  "Help",
  "More",
  "No",
  "Please",
  "Sad",
  "Thank You",
  "Water",
  "Yes",
];

type PredictionPanelProps = {
  isCommunicating: boolean;
  statusText: string;
  prediction: string;
  /** Latched sign confidence (0–100), shown under the word; persists until a new high-conf sign */
  confidencePercent: number;
  /** Live top-1 confidence (0–100) for the bar; defaults to confidencePercent */
  barPercent?: number;
  stabilizing?: boolean;
  debugConfidencePercent?: number | null;
  debugRawPrediction?: string | null;
  /** Resets UI and calls backend reset_ui_only when provided */
  onResetSession?: () => void;
};

/** Word shown in the large Current Sign label — matches on-screen display logic. */
export function getVisibleSignWord({
  statusText,
  prediction,
  debugRawPrediction,
  isCommunicating,
  stabilizing,
}: Pick<
  PredictionPanelProps,
  "statusText" | "prediction" | "debugRawPrediction" | "isCommunicating" | "stabilizing"
>): string {
  const showConfidence = !statusText && !!prediction;
  if (showConfidence) {
    return String(prediction).trim();
  }
  if (isCommunicating && stabilizing && debugRawPrediction) {
    return String(debugRawPrediction).trim();
  }
  return "";
}

export const PredictionPanel = ({
  isCommunicating,
  statusText,
  prediction,
  confidencePercent,
  barPercent,
  stabilizing = false,
  debugConfidencePercent = null,
  debugRawPrediction = null,
  onResetSession,
}: PredictionPanelProps) => {
  const navigate = useNavigate();
  const ACTIVATION_THRESHOLD = 55;
  const displayText =
    statusText ||
    prediction ||
    (isCommunicating
      ? "Connecting to AI..."
      : 'Click "Start Communicating" to begin.');

  const showConfidence = !statusText && !!prediction;
  const visibleWord = getVisibleSignWord({
    statusText,
    prediction,
    debugRawPrediction,
    isCommunicating,
    stabilizing,
  });
  const hasVisibleWord = visibleWord.length > 0;
  const [confirmedLocked, setConfirmedLocked] = useState(false);
  const isConfirmed = !!prediction && (confirmedLocked || confidencePercent >= ACTIVATION_THRESHOLD);
  const barW = Math.max(0, Math.min(100, barPercent ?? confidencePercent));

  const handleReset = useCallback(() => {
    onResetSession?.();
  }, [onResetSession]);

  const handleCopy = useCallback(async () => {
    if (!hasVisibleWord) return;
    const text = visibleWord.toUpperCase();
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied to clipboard");
    } catch {
      toast.error("Could not copy to clipboard");
    }
  }, [hasVisibleWord, visibleWord]);

  const handleTranslate = useCallback(() => {
    if (!hasVisibleWord) return;
    navigate("/translator", {
      state: { text: visibleWord.toUpperCase() },
    });
  }, [hasVisibleWord, visibleWord, navigate]);

  useEffect(() => {
    if (!onResetSession) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "r" || e.key === "R" || e.key === "Escape") {
        e.preventDefault();
        onResetSession();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onResetSession]);

  useEffect(() => {
    if (!prediction) {
      setConfirmedLocked(false);
      return;
    }
    if (confidencePercent >= ACTIVATION_THRESHOLD) {
      setConfirmedLocked(true);
    }
  }, [prediction, confidencePercent]);

  return (
    <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6 h-full">
      <div className="flex flex-col gap-3 mb-4">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-xl font-semibold text-white">Current Sign</h2>
          <div className="flex items-center gap-2 shrink-0">
            {onResetSession && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleReset}
                className="text-xs h-8 border-white/20 text-gray-200 hover:bg-white/10"
                title="Reset session (R or Esc)"
              >
                <RotateCcw className="w-3.5 h-3.5 mr-1" />
                Reset Session
              </Button>
            )}
            {stabilizing && (
              <div className="text-xs font-semibold text-gray-200 bg-white/10 border border-white/10 px-2 py-1 rounded-full">
                Stabilizing...
              </div>
            )}
            {isConfirmed && (
              <div className="text-xs font-semibold text-emerald-200 bg-emerald-500/20 border border-emerald-400/40 px-2 py-1 rounded-full inline-flex items-center gap-1">
                <BadgeCheck className="w-3.5 h-3.5" />
                Sign Detected
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-300">
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${
              isCommunicating ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)] animate-pulse" : "bg-sky-400"
            }`}
            aria-hidden
          />
          <span className="font-medium">
            {isCommunicating ? "Processing…" : "Ready"}
          </span>
          <span className="text-gray-500 hidden sm:inline">
            {isCommunicating
              ? "AI is receiving frames"
              : "Last sign stays until you reset or show a new one (>15% conf)"}
          </span>
        </div>
      </div>

      <div className="space-y-4">
        <div className="p-4 bg-white/10 rounded-lg min-h-[112px] flex items-center justify-center text-center">
          {showConfidence ? (
            <div>
              <div
                className={`text-4xl md:text-5xl font-black tracking-wide ${
                  isConfirmed ? "text-emerald-400 animate-pulse" : "text-white"
                }`}
              >
                {String(prediction).toUpperCase()}
              </div>
              <div className="text-sm text-gray-300 mt-1">
                Confidence: {confidencePercent}%
              </div>
            </div>
          ) : (
            <div>
              <p className="text-lg text-gray-200 font-semibold">{displayText}</p>
              {debugConfidencePercent !== null && Number.isFinite(debugConfidencePercent) && (
                <p className="text-sm text-gray-300 mt-1">Confidence: {Math.round(debugConfidencePercent)}%</p>
              )}
              {debugRawPrediction && (
                <p className="text-4xl md:text-5xl font-black text-emerald-400 mt-2 tracking-wide">
                  {String(debugRawPrediction).toUpperCase()}
                </p>
              )}
            </div>
          )}
        </div>

        {hasVisibleWord && (
          <div className="flex flex-wrap justify-center gap-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void handleCopy()}
              className="border-white/20 text-gray-200 hover:bg-white/10"
            >
              <Copy className="w-4 h-4 mr-2" />
              Copy
            </Button>
            <Button
              type="button"
              variant="hero"
              size="sm"
              onClick={handleTranslate}
              className="min-w-[120px]"
            >
              <Languages className="w-4 h-4 mr-2" />
              Translate
            </Button>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Live signal (×100 from model)</span>
            <span className="text-white font-bold tabular-nums">{barW}%</span>
          </div>
          <div className="bg-white/10 rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-ai-secondary to-ai-accent rounded-full"
              style={{
                width: `${barW}%`,
                transition: "width 0.3s ease-in-out",
              }}
            />
          </div>
        </div>

        <div className="text-sm text-gray-300 space-y-1">
          <p>
            Tips: keep your hand centered, good lighting, and hold the sign steady
            for ~1–2 seconds.
          </p>
          {onResetSession && (
            <p className="text-xs text-gray-400">Keyboard: R or Esc to reset session.</p>
          )}
        </div>
      </div>
    </Card>
  );
};

type TextToSignDictionaryProps = {
  className?: string;
};

/**
 * Text-to-Sign dictionary: loads the first .mov for a word from the backend
 * GET /get_sign_video/<word> (served from SignEase_Project/SignEase_dataset).
 */
export const TextToSignDictionary = ({ className = "" }: TextToSignDictionaryProps) => {
  const [searchTerm, setSearchTerm] = useState("Hello");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resolvedBase, setResolvedBase] = useState<string | null>(null);
  const [candidateBases, setCandidateBases] = useState<string[]>([]);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [activeQuery, setActiveQuery] = useState("");

  const loadVideoForQuery = (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    setLoadError(null);
    setIsLoading(true);
    setResolvedBase(null);
    setActiveQuery(trimmed);
    setCandidateBases([...API_BASE_CANDIDATES]);
    setCandidateIndex(0);
    const firstBase = API_BASE_CANDIDATES[0];
    setVideoUrl(`${firstBase}/get_sign_video/${encodeURIComponent(trimmed)}`);
  };

  const applySearch = () => loadVideoForQuery(searchTerm);

  return (
    <Card className={`border-white/10 bg-white/5 backdrop-blur-md p-6 ${className}`}>
      <h2 className="text-xl font-semibold text-white mb-4">Text-to-Sign dictionary</h2>

      <div className="space-y-4">
        <div>
          <label className="text-sm text-gray-300 block mb-2">Search a supported word</label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && applySearch()}
              className="flex-1 bg-white/10 border border-white/20 text-white rounded px-3 py-2 placeholder:text-gray-500"
              placeholder="e.g. Thank You"
            />
            <Button type="button" variant="hero" onClick={applySearch} className="shrink-0">
              <Search className="w-4 h-4 mr-2" />
              Show sign
            </Button>
          </div>
        </div>

        <div>
          <p className="text-sm text-gray-300 mb-2">Supported vocabulary</p>
          <div className="flex flex-wrap gap-2">
            {SUPPORTED_VOCABULARY.map((w) => (
              <button
                key={w}
                type="button"
                onClick={() => {
                  setSearchTerm(w);
                  void loadVideoForQuery(w);
                }}
                className="text-xs text-gray-200 bg-white/10 hover:bg-white/15 border border-white/10 px-2 py-1 rounded transition-colors"
              >
                {w}
              </button>
            ))}
          </div>
        </div>

        {videoUrl && (
          <div className="rounded-lg overflow-hidden border border-white/10 bg-black">
            <video
              key={videoUrl}
              className="w-full max-h-[360px] object-contain"
              src={videoUrl}
              controls
              autoPlay
              muted
              loop
              playsInline
              onLoadedData={() => {
                setIsLoading(false);
                const base = candidateBases[candidateIndex] ?? null;
                setResolvedBase(base);
                setLoadError(null);
              }}
              onError={() => {
                const nextIdx = candidateIndex + 1;
                if (nextIdx < candidateBases.length) {
                  setCandidateIndex(nextIdx);
                  const nextBase = candidateBases[nextIdx];
                  setVideoUrl(`${nextBase}/get_sign_video/${encodeURIComponent(activeQuery)}`);
                  return;
                }
                setIsLoading(false);
                setLoadError("Could not load a reference clip. Start backend2 API and try again.");
              }}
            />
            {resolvedBase && (
              <p className="text-center text-xs text-gray-400 py-1">Source: {resolvedBase}</p>
            )}
            {isLoading && (
              <p className="text-center text-sm text-gray-400 py-2">Loading…</p>
            )}
            {loadError && <p className="text-center text-sm text-red-300 py-2">{loadError}</p>}
          </div>
        )}
        {!videoUrl && loadError && <p className="text-center text-sm text-red-300 py-2">{loadError}</p>}
      </div>
    </Card>
  );
};
