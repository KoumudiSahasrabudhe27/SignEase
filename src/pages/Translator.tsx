import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Languages, ArrowRight, ArrowLeft } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { LANGUAGES, SOURCE_LANGUAGE } from "@/constants/languages";
import { translateText } from "@/lib/translationApi";

type TranslatorLocationState = {
  text?: string;
};

const Translator = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [targetLanguage, setTargetLanguage] = useState<string>("hin_Deva");
  const [isTranslating, setIsTranslating] = useState(false);

  useEffect(() => {
    const state = location.state as TranslatorLocationState | null;
    const text = state?.text?.trim();
    if (text) {
      setInputText(text);
      setOutputText("");
    }
  }, [location.state]);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;

    setIsTranslating(true);
    setOutputText("");

    try {
      const result = await translateText({
        text: inputText,
        source_lang: SOURCE_LANGUAGE.code,
        target_lang: targetLanguage as (typeof LANGUAGES)[number]["code"],
      });
      setOutputText(result.translation);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Translation failed. Please try again.";
      toast.error(message);
    } finally {
      setIsTranslating(false);
    }
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
              <p className="text-gray-300">
                Translate English to Indian languages using AI4Bharat IndicTrans2
              </p>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-lg font-medium text-gray-300 mb-3">
                Source Language
              </label>
              <Select value={SOURCE_LANGUAGE.code} disabled>
                <SelectTrigger className="bg-white/10 border-white/20 text-white h-12 text-base">
                  <SelectValue>{SOURCE_LANGUAGE.name}</SelectValue>
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-white/20">
                  <SelectItem
                    value={SOURCE_LANGUAGE.code}
                    className="text-white hover:bg-white/10"
                  >
                    {SOURCE_LANGUAGE.name}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-lg font-medium text-gray-300 mb-3">
                Target Language
              </label>
              <Select value={targetLanguage} onValueChange={setTargetLanguage}>
                <SelectTrigger className="bg-white/10 border-white/20 text-white h-12 text-base">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-white/20 max-h-72">
                  {LANGUAGES.map((lang) => (
                    <SelectItem
                      key={lang.code}
                      value={lang.code}
                      className="text-white hover:bg-white/10"
                    >
                      {lang.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-6 mb-8">
            <div>
              <label className="block text-lg font-medium text-gray-300 mb-3">
                Input Text
              </label>
              <Textarea
                placeholder="Enter English text to translate..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="bg-white/10 border-white/20 text-white placeholder:text-gray-400 min-h-[180px] text-base"
              />
            </div>

            <div>
              <label className="block text-lg font-medium text-gray-300 mb-3">
                Output Translation
              </label>
              <Textarea
                placeholder="Translation will appear here..."
                value={outputText}
                readOnly
                className="bg-white/10 border-white/20 text-white placeholder:text-gray-400 min-h-[180px] text-base"
              />
            </div>
          </div>

          <div className="flex justify-center">
            <Button
              variant="hero"
              onClick={handleTranslate}
              disabled={!inputText.trim() || isTranslating}
              className="min-w-[160px] h-12 text-base"
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
