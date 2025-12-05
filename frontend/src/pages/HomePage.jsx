import { useNavigate } from "react-router-dom";

const HomePage = () => {
  const navigate = useNavigate();

  const modules = [
    {
      id: 1,
      title: "Bone Density Analysis",
      icon: "🦴",
      description:
        "Analyze bone mineral density from X-ray images using advanced AI models",
      path: "/bone-density",
      gradient: "from-blue-500 to-cyan-500",
    },
    {
      id: 2,
      title: "Fracture Detection",
      icon: "🩻",
      description:
        "Multi-transformer ensemble AI system for accurate fracture detection with explainability",
      path: "/fracture-detection",
      gradient: "from-purple-500 to-pink-500",
    },
    // {
    //   id: 3,
    //   title: "Model Performance",
    //   icon: "📊",
    //   description:
    //     "Comprehensive evaluation metrics, confusion matrices, ROC curves, and model comparisons",
    //   path: "/model-performance",
    //   gradient: "from-green-500 to-teal-500",
    // },
    // {
    //   id: 4,
    //   title: "Report Page",
    //   icon: "📊",
    //   description:
    //     "Comprehensive evaluation metrics, confusion matrices, ROC curves, and model comparisons",
    //   path: "/report",
    //   gradient: "from-green-500 to-teal-500",
    // },
  ];

  return (
    <div className="min-h-screen p-8">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-12">
        <div className="text-center mb-8">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            🏥 Bone Health AI Platform
          </h1>
          <p className="text-xl text-gray-600">
            Advanced Medical Imaging Analysis powered by Transformer Models
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-2">
                Welcome to Bone Health Analysis
              </h2>
              <p className="text-gray-600">
                Select a module below to begin your analysis journey
              </p>
            </div>
            <div className="text-6xl">🎯</div>
          </div>
        </div>
      </div>

      {/* Module Cards */}
      <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-8">
        {modules.map((module) => (
          <div
            key={module.id}
            onClick={() => navigate(module.path)}
            className="group cursor-pointer"
          >
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden transition-all duration-300 hover:shadow-2xl hover:-translate-y-2">
              {/* Gradient Header */}
              <div
                className={`h-32 bg-gradient-to-br ${module.gradient} flex items-center justify-center`}
              >
                <span className="text-8xl filter drop-shadow-lg">
                  {module.icon}
                </span>
              </div>

              {/* Content */}
              <div className="p-6">
                <h3 className="text-2xl font-bold text-gray-800 mb-3 group-hover:text-purple-600 transition-colors">
                  {module.title}
                </h3>
                <p className="text-gray-600 mb-4 min-h-[60px]">
                  {module.description}
                </p>

                <button className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-3 rounded-xl font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all shadow-md hover:shadow-lg">
                  Launch Module →
                </button>
              </div>

              {/* Module Badge */}
              <div className="px-6 pb-4">
                <div className="inline-block bg-gray-100 px-4 py-2 rounded-full text-sm font-semibold text-gray-700">
                  Module {module.id}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto mt-16">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl shadow-2xl p-8 text-white">
          <h3 className="text-3xl font-bold mb-6 text-center">
            Platform Features
          </h3>
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-5xl mb-3">🤖</div>
              <h4 className="font-bold mb-2">AI-Powered</h4>
              <p className="text-sm text-blue-100">
                Multi-transformer ensemble models
              </p>
            </div>
            <div className="text-center">
              <div className="text-5xl mb-3">📈</div>
              <h4 className="font-bold mb-2">High Accuracy</h4>
              <p className="text-sm text-blue-100">
                96%+ accuracy with AUC 0.98
              </p>
            </div>
            <div className="text-center">
              <div className="text-5xl mb-3">🔍</div>
              <h4 className="font-bold mb-2">Explainable AI</h4>
              <p className="text-sm text-blue-100">
                Grad-CAM & attention visualization
              </p>
            </div>
            <div className="text-center">
              <div className="text-5xl mb-3">⚡</div>
              <h4 className="font-bold mb-2">Real-time</h4>
              <p className="text-sm text-blue-100">
                Fast inference and results
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="max-w-7xl mx-auto mt-12 text-center text-gray-500 text-sm">
        <p>© 2025 Bone Health AI Platform • Powered by Transformer Models</p>
      </div>
    </div>
  );
};

export default HomePage;
