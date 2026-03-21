import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";

type PredictionPanelProps = {
  isCommunicating: boolean;
  statusText: string;
  prediction: string;
  confidencePercent: number;
  stabilizing?: boolean;
  debugConfidencePercent?: number | null;
  debugRawPrediction?: string | null;
};

export const PredictionPanel = ({
  isCommunicating,
  statusText,
  prediction,
  confidencePercent,
  stabilizing = false,
  debugConfidencePercent = null,
  debugRawPrediction = null,
}: PredictionPanelProps) => {
  const displayText =
    statusText ||
    prediction ||
    (isCommunicating
      ? "Connecting to AI..."
      : 'Click "Start Communicating" to begin.');

  const showConfidence = !statusText && !!prediction;

  return (
    <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6 h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white">Live Prediction</h2>
        {stabilizing && (
          <div className="text-xs font-semibold text-gray-200 bg-white/10 border border-white/10 px-2 py-1 rounded-full">
            Stabilizing...
          </div>
        )}
      </div>

      <div className="space-y-4">
        <div className="p-4 bg-white/10 rounded-lg min-h-[112px] flex items-center justify-center text-center">
          {showConfidence ? (
            <div>
              <div className="text-3xl font-bold text-white">
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
                <p className="text-sm text-gray-300 mt-1">Latest: {String(debugRawPrediction).toUpperCase()}</p>
              )}
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Confidence</span>
            <span className="text-white font-semibold">{confidencePercent}%</span>
          </div>
          <div className="bg-white/10 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(0, Math.min(100, confidencePercent))}%` }}
              transition={{ duration: 0.6 }}
              className="h-full bg-gradient-to-r from-ai-secondary to-ai-accent rounded-full"
            />
          </div>
        </div>

        <div className="text-sm text-gray-300">
          Tips: keep your hand centered, good lighting, and hold the sign steady
          for ~1–2 seconds.
        </div>
      </div>
    </Card>
  );
};

