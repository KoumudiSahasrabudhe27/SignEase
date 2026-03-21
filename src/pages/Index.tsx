import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FeatureCard } from "@/components/FeatureCard";
import { Mic, Camera, Type, Info, Brain, Accessibility, Languages, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: "Speech → Sign",
      description: "Convert spoken words into sign language representations using advanced speech recognition and AI translation.",
      icon: Mic,
      href: "/speech-to-sign",
      gradient: "from-blue-500 to-purple-600"
    },
    {
      title: "Sign → Text",
      description: "Use your camera to capture sign language and convert it to readable text with real-time AI analysis.",
      icon: Camera,
      href: "/sign-to-text",
      gradient: "from-purple-600 to-pink-600"
    },
    {
      title: "Text → Sign",
      description: "Transform written text into sign language symbols for easy communication and learning.",
      icon: Type,
      href: "/text-to-sign",
      gradient: "from-pink-600 to-red-500"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 px-4">
        <div className="absolute inset-0 bg-gradient-to-br from-ai-primary/20 via-ai-secondary/20 to-ai-accent/20" />
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-ai-primary/10 rounded-full blur-3xl animate-float" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-ai-secondary/10 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
        </div>
        
        <div className="relative max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="p-3 rounded-full bg-gradient-to-r from-ai-primary to-ai-secondary">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-white via-white to-gray-300 bg-clip-text text-transparent">
                Smart Sign Language
              </h1>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Interpretation using AI
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
              Break down communication barriers with our advanced AI-powered platform. 
              Convert between speech, text, and sign language seamlessly for inclusive communication.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="flex flex-col sm:flex-row gap-4 justify-center mb-12"
          >
            <Button
              variant="hero"
              size="lg"
              className="text-lg px-8 py-4"
              onClick={() => navigate("/sign-to-text")}
            >
              <Accessibility className="w-5 h-5" />
              Start Communicating
            </Button>
            <Button variant="glass" size="lg" className="text-lg px-8 py-4" onClick={() => window.location.href = "/about"}>
              <Info className="w-5 h-5" />
              Learn More
            </Button>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Core Features</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Choose from our three main translation modules to facilitate seamless communication
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="grid md:grid-cols-3 gap-8 mb-16"
          >
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.7 + index * 0.1 }}
              >
                <FeatureCard {...feature} />
              </motion.div>
            ))}
          </motion.div>

          {/* Translator Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6 text-center">
              <div className="flex items-center justify-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-gradient-to-r from-ai-secondary to-ai-accent">
                  <Languages className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white">Indian Language Translator</h3>
              </div>
              <p className="text-gray-300 mb-6">
                Translate English text to various Indian languages with our advanced AI translator
              </p>
              <Button 
                variant="hero" 
                onClick={() => window.location.href = "/translator"}
                className="px-8 py-3"
              >
                Open Translator
                <ArrowRight className="w-5 h-5" />
              </Button>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-white/10">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-gray-400">
            Making communication accessible for everyone through the power of AI
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
