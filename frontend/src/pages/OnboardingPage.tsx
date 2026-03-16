import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface OnboardingStep {
  title: string;
  description: string;
  icon: string;
}

const steps: OnboardingStep[] = [
  {
    title: 'Welcome to DAPPY',
    description: 'Your personal cognitive memory assistant. I help you remember important information and recall it when you need it.',
    icon: '🧠',
  },
  {
    title: 'Tell Me Anything',
    description: 'Share facts about your life, preferences, goals, or anything you want me to remember. The more you share, the better I can assist you.',
    icon: '💬',
  },
  {
    title: 'I Remember Everything',
    description: 'I organize your memories intelligently using a knowledge graph, making it easy to recall information later through natural conversation.',
    icon: '🔗',
  },
  {
    title: 'Ask Me Anytime',
    description: 'Just chat with me naturally. Ask questions about what you\'ve told me, and I\'ll help you recall the information you need.',
    icon: '💡',
  },
];

export default function OnboardingPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const navigate = useNavigate();

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      localStorage.setItem('dappy_onboarding_complete', 'true');
      navigate('/chat');
    }
  };

  const handleSkip = () => {
    localStorage.setItem('dappy_onboarding_complete', 'true');
    navigate('/chat');
  };

  const step = steps[currentStep];

  return (
    <div className="min-h-full flex items-center justify-center bg-gradient-to-br from-surface-900 via-surface-800 to-surface-950 px-6">
      <div className="w-full max-w-2xl">
        {/* Progress indicators */}
        <div className="flex justify-center gap-2 mb-12">
          {steps.map((_, index) => (
            <div
              key={index}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                index === currentStep
                  ? 'w-12 bg-primary-500'
                  : index < currentStep
                  ? 'w-8 bg-primary-600/50'
                  : 'w-8 bg-surface-700'
              }`}
            />
          ))}
        </div>

        {/* Content card */}
        <div className="bg-surface-800/60 backdrop-blur-xl border border-surface-700/50 rounded-3xl p-12 shadow-2xl text-center">
          {/* Icon */}
          <div className="inline-flex items-center justify-center w-24 h-24 rounded-3xl bg-primary-600/20 mb-8 text-6xl">
            {step.icon}
          </div>

          {/* Title */}
          <h1 className="text-3xl font-bold text-white mb-4 tracking-tight">
            {step.title}
          </h1>

          {/* Description */}
          <p className="text-surface-300 text-lg leading-relaxed max-w-lg mx-auto mb-12">
            {step.description}
          </p>

          {/* Actions */}
          <div className="flex gap-4 justify-center">
            {currentStep < steps.length - 1 && (
              <button
                onClick={handleSkip}
                className="px-6 py-3 rounded-xl text-surface-400 hover:text-surface-200 font-medium transition-colors"
              >
                Skip
              </button>
            )}
            <button
              onClick={handleNext}
              className="px-8 py-3 rounded-xl bg-primary-600 hover:bg-primary-500 text-white font-semibold shadow-lg shadow-primary-600/25 hover:shadow-primary-500/30 transition-all"
            >
              {currentStep < steps.length - 1 ? 'Next' : 'Get Started'}
            </button>
          </div>

          {/* Step counter */}
          <p className="text-surface-500 text-sm mt-8">
            {currentStep + 1} of {steps.length}
          </p>
        </div>
      </div>
    </div>
  );
}
