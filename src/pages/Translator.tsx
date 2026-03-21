import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Languages, ArrowRight, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";

const indianLanguages = [
  { code: "hi", name: "Hindi" },
  { code: "bn", name: "Bengali" },
  { code: "te", name: "Telugu" },
  { code: "mr", name: "Marathi" },
  { code: "ta", name: "Tamil" },
  { code: "gu", name: "Gujarati" },
  { code: "kn", name: "Kannada" },
  { code: "ml", name: "Malayalam" },
  { code: "pa", name: "Punjabi" },
  { code: "or", name: "Odia" },
];

const Translator = () => {
  const navigate = useNavigate();
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("hi");
  const [isTranslating, setIsTranslating] = useState(false);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    
    setIsTranslating(true);
    // Simulate translation - in real app, connect to translation API
    setTimeout(() => {
      setOutputText(`[Translated to ${indianLanguages.find(l => l.code === selectedLanguage)?.name}]: ${inputText}`);
      setIsTranslating(false);
    }, 1500);
  };

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
          <h1 className="text-3xl font-bold text-white">Indian Language Translator</h1>
        </div>

        <Card className="border-white/10 bg-white/5 backdrop-blur-md p-8">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 rounded-lg bg-gradient-to-r from-ai-secondary to-ai-accent">
              <Languages className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-white">Language Translation</h2>
              <p className="text-gray-300">Translate English text to various Indian languages</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-lg font-medium text-gray-300 mb-3">
                English Text
              </label>
              <Textarea
                placeholder="Enter text to translate..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="bg-white/10 border-white/20 text-white placeholder:text-gray-400 min-h-[200px] text-base"
              />
            </div>

            <div>
              <label className="block text-lg font-medium text-gray-300 mb-3">
                Translation
              </label>
              <Textarea
                placeholder="Translation will appear here..."
                value={outputText}
                readOnly
                className="bg-white/10 border-white/20 text-white placeholder:text-gray-400 min-h-[200px] text-base"
              />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-6 items-end">
            <div className="flex-1">
              <label className="block text-lg font-medium text-gray-300 mb-3">
                Target Language
              </label>
              <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                <SelectTrigger className="bg-white/10 border-white/20 text-white h-12 text-base">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-white/20">
                  {indianLanguages.map((lang) => (
                    <SelectItem key={lang.code} value={lang.code} className="text-white hover:bg-white/10">
                      {lang.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              variant="hero"
              onClick={handleTranslate}
              disabled={!inputText.trim() || isTranslating}
              className="min-w-[140px] h-12 text-base"
            >
              {isTranslating ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                />
              ) : (
                <>
                  Translate
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </Button>
          </div>
        </Card>
      </motion.div>
    </div>
  );
};

export default Translator;