import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Brain, Users, Accessibility, Zap } from "lucide-react";
import { useNavigate } from "react-router-dom";

const About = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Recognition",
      description: "Advanced machine learning algorithms for accurate sign language interpretation and generation."
    },
    {
      icon: Users,
      title: "Inclusive Communication",
      description: "Breaking down barriers between hearing and deaf communities through seamless translation."
    },
    {
      icon: Accessibility,
      title: "Universal Access",
      description: "Making digital content accessible to everyone, regardless of hearing ability."
    },
    {
      icon: Zap,
      title: "Real-time Processing",
      description: "Instant conversion between speech, text, and sign language for fluid communication."
    }
  ];

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
          <h1 className="text-3xl font-bold text-white">About Our Technology</h1>
        </div>

        <div className="space-y-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="border-white/10 bg-white/5 backdrop-blur-md p-8">
              <h2 className="text-2xl font-bold text-white mb-4">
                Smart Sign Language Interpretation
              </h2>
              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                Our platform leverages cutting-edge artificial intelligence to bridge the communication gap 
                between hearing and deaf communities. By combining computer vision, natural language processing, 
                and machine learning, we provide seamless translation between spoken language, text, and sign language.
              </p>
              <p className="text-gray-300 leading-relaxed">
                The system supports multiple input methods including speech recognition, webcam-based sign detection, 
                and text input, making it versatile for various communication scenarios. Our AI models are trained 
                on diverse datasets to ensure accuracy across different signing styles and regional variations.
              </p>
            </Card>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + index * 0.1 }}
              >
                <Card className="border-white/10 bg-white/5 backdrop-blur-md p-6 h-full">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 rounded-lg bg-gradient-to-r from-ai-primary to-ai-secondary">
                      <feature.icon className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-semibold text-white">{feature.title}</h3>
                  </div>
                  <p className="text-gray-300">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <Card className="border-white/10 bg-white/5 backdrop-blur-md p-8">
              <h2 className="text-2xl font-bold text-white mb-4">How It Works</h2>
              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-ai-primary to-ai-secondary flex items-center justify-center text-white font-bold">1</div>
                  <div>
                    <h3 className="text-white font-semibold">Input Processing</h3>
                    <p className="text-gray-300">The system captures input through speech, camera, or text interface.</p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-ai-secondary to-ai-accent flex items-center justify-center text-white font-bold">2</div>
                  <div>
                    <h3 className="text-white font-semibold">AI Analysis</h3>
                    <p className="text-gray-300">Advanced algorithms analyze and interpret the input using trained models.</p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-ai-accent to-ai-primary flex items-center justify-center text-white font-bold">3</div>
                  <div>
                    <h3 className="text-white font-semibold">Output Generation</h3>
                    <p className="text-gray-300">The system generates appropriate output in the target format (text, signs, or speech).</p>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="text-center"
          >
            <Card className="border-white/10 bg-white/5 backdrop-blur-md p-8">
              <h2 className="text-2xl font-bold text-white mb-4">Our Mission</h2>
              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                To create a world where communication barriers don't exist, enabling everyone to express 
                themselves freely and understand each other regardless of hearing ability.
              </p>
              <Button variant="hero" onClick={() => navigate("/")} className="text-lg px-8 py-3">
                Start Using the Platform
              </Button>
            </Card>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default About;