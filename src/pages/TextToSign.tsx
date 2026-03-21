import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Type, Wand2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const WORDS_10 = ["food", "hello", "help", "more", "no", "please", "sad", "thank you", "water", "yes"];

const TextToSign = () => {
  const navigate = useNavigate();
  const [inputText, setInputText] = useState("");
  const [signOutput, setSignOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedWord, setSelectedWord] = useState("hello");
  const [selectedVideoUrl, setSelectedVideoUrl] = useState("");
  const [selectedVideoPath, setSelectedVideoPath] = useState("");

  const generateSigns = async () => {
    setIsGenerating(true);
    try {
      const resp = await fetch(`http://localhost:8001/get_sign_video/${encodeURIComponent(selectedWord)}`);
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setSelectedVideoPath(String(data.file_path ?? ""));
      setSelectedVideoUrl(`http://localhost:8001${String(data.video_url ?? "")}`);
      setSignOutput(`Selected word: ${selectedWord}`);
    } catch (err) {
      console.error(err);
      setSignOutput("(error loading sign video)");
      setSelectedVideoPath("");
      setSelectedVideoUrl("");
    } finally {
      setIsGenerating(false);
    }
  };


//   const generateSigns = async () => {
//     if (!inputText.trim()) return;

//     setIsGenerating(true);
//     try {
//       const resp = await fetch("http://localhost:8000/api/text-to-sign", {
//         method: "POST",
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ text: inputText })
//       });
//       if (!resp.ok) throw new Error(`Error: ${resp.status}`);
//       const data = await resp.json();
//       setSignOutput(data.signs); // e.g. "🤟 👋 🙏 ..."
//     } catch (err) {
//       console.error(err);
//       setSignOutput("(error generating signs)");
//     } finally {
//       setIsGenerating(false);
//     }
//   };


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
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Type className="w-5 h-5" />
              Text Input
            </h2>
            
            <div className="space-y-4">
              <label className="text-sm text-gray-300">Select dictionary word</label>
              <select
                value={selectedWord}
                onChange={(e) => setSelectedWord(e.target.value)}
                className="bg-white/10 border border-white/20 text-white rounded px-3 py-2"
              >
                {WORDS_10.map((w) => (
                  <option key={w} value={w} className="text-black">
                    {w}
                  </option>
                ))}
              </select>

              <Textarea
                placeholder="Enter text to convert to sign language..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="bg-white/10 border-white/20 text-white placeholder:text-gray-400 min-h-[120px] text-lg"
              />
              
              <Button
                variant="hero"
                onClick={generateSigns}
                disabled={isGenerating}
                className="w-full"
              >
                {isGenerating ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"
                    />
                    Generating Signs...
                  </>
                ) : (
                  <>
                    <Wand2 className="w-4 h-4" />
                    Convert to Sign Language
                  </>
                )}
              </Button>
            </div>
          </Card>

          {signOutput && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6">
                <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                  <span className="text-2xl">🤟</span>
                  Sign Language Output
                </h2>
                
                <div className="space-y-4">
                  <div className="p-6 bg-white/10 rounded-lg text-center">
                    <div className="text-6xl mb-4 leading-relaxed">
                      {signOutput}
                    </div>
                  </div>
                  {selectedVideoUrl && (
                    <video
                      controls
                      autoPlay
                      muted
                      loop
                      className="w-full rounded-lg border border-white/10 bg-black"
                      src={selectedVideoUrl}
                    />
                  )}
                  
                  <div className="text-center">
                    <p className="text-gray-300 mb-4">Original text: "{inputText}"</p>
                    {selectedVideoPath && (
                      <p className="text-gray-400 text-xs break-all mb-4">Dataset file: {selectedVideoPath}</p>
                    )}
                    <Button variant="glass" onClick={() => navigator.clipboard.writeText(signOutput)}>
                      Copy Signs
                    </Button>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6">
            <h3 className="text-lg font-semibold text-white mb-3">Supported Words</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              {WORDS_10.map((word) => (
                <span key={word} className="text-gray-300 bg-white/5 px-2 py-1 rounded">
                  {word}
                </span>
              ))}
            </div>
          </Card>
        </div>
      </motion.div>
    </div>
  );
};

export default TextToSign;