import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
  href: string;
  gradient: string;
}

export const FeatureCard = ({ title, description, icon: Icon, href, gradient }: FeatureCardProps) => {
  return (
    <motion.div
      whileHover={{ scale: 1.02, y: -5 }}
      transition={{ duration: 0.3 }}
      className="group"
    >
      <Card className="relative overflow-hidden border-white/10 bg-white/5 backdrop-blur-md h-full">
        <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-10 group-hover:opacity-20 transition-opacity duration-300`} />
        <div className="relative p-6 flex flex-col h-full">
          <div className="flex items-center gap-3 mb-4">
            <div className={`p-3 rounded-lg bg-gradient-to-r ${gradient}`}>
              <Icon className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-white">{title}</h3>
          </div>
          <p className="text-gray-300 mb-6 flex-grow">{description}</p>
          <Button 
            variant="glass" 
            className="self-start group-hover:bg-white/20"
            onClick={() => window.location.href = href}
          >
            Try Feature
          </Button>
        </div>
      </Card>
    </motion.div>
  );
};