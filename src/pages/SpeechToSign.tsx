import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Mic, Square, Volume2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const SpeechToSign = () => {
  const navigate = useNavigate();
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [signOutput, setSignOutput] = useState("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach(track => track.stop()); // Stop the microphone stream

        // Now, send the audioBlob to the backend
        setTranscript("Processing audio..."); // Indicate processing to the user
        setSignOutput("");
        try {
          const formData = new FormData();
          formData.append("file", audioBlob, "speech.webm"); // Use "file" as the key matching backend

          const response = await fetch("http://localhost:8000/api/speech-to-sign", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
          }
          const data = await response.json();
          setTranscript(data.transcript);
          setSignOutput(data.signOutput);
        } catch (err) {
          console.error("Error sending audio to backend:", err);
          setTranscript("(Error processing speech)");
          setSignOutput("");
        } finally {
          setIsRecording(false); // Set to false after upload/processing is complete
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setTranscript("Recording...");
      setSignOutput("");
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setIsRecording(false);
      setTranscript("(Microphone access denied or error)");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      // setIsRecording(false) will be handled in onstop after processing
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
          <h1 className="text-3xl font-bold text-white">Speech to Sign Language</h1>
        </div>

        <div className="grid gap-6">
          <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Volume2 className="w-5 h-5" />
              Speech Input
            </h2>
            
            <div className="flex flex-col items-center gap-6">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={isRecording ? stopRecording : startRecording}
                className={`w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isRecording 
                    ? "bg-red-500 animate-pulse" 
                    : "bg-gradient-to-r from-ai-primary to-ai-secondary hover:shadow-lg hover:shadow-ai-primary/25"
                }`}
              >
                {isRecording ? (
                  <Square className="w-8 h-8 text-white" />
                ) : (
                  <Mic className="w-8 h-8 text-white" />
                )}
              </motion.button>
              
              <p className="text-gray-300 text-center">
                {isRecording ? "Recording... Click to stop" : "Click to start recording"}
              </p>
            </div>

            {transcript && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 p-4 bg-white/10 rounded-lg"
              >
                <h3 className="text-white font-medium mb-2">Transcript:</h3>
                <p className="text-gray-300">{transcript}</p>
              </motion.div>
            )}
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
                
                <div className="text-center">
                  <div className="text-6xl mb-4 p-8 bg-white/10 rounded-lg">
                    {signOutput}
                  </div>
                  <p className="text-gray-300">Generated sign language representation</p>
                </div>
              </Card>
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default SpeechToSign;