import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Type } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { TextToSignDictionary } from "@/components/signease/PredictionPanel";

const TextToSign = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background p-4">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="max-w-4xl mx-auto"
      >
        <div className="flex items-center gap-4 mb-8">
          <Button variant="ghost" onClick={() => navigate("/")} className="text-white">
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Button>
          <h1 className="text-3xl font-bold text-white">Text to Sign Language</h1>
        </div>

        <div className="grid gap-6">
          <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6">
            <h2 className="text-xl font-semibold text-white mb-2 flex items-center gap-2">
              <Type className="w-5 h-5" />
              Dictionary lookup
            </h2>
            <p className="text-sm text-gray-300 mb-4">
              Search or tap a word to play a reference clip from the local 10-word dataset (served by the
              Flask API at <span className="text-white/90">/get_sign_video/&lt;word&gt;</span>).
            </p>
          </Card>

          <TextToSignDictionary />
        </div>
      </motion.div>
    </div>
  );
};

export default TextToSign;
