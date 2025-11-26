import { useState } from "react";
import ModuleSelector from "./components/ModuleSelector";
import FractureDetection from "./components/FractureDetection";
import BoneHealth from "./components/BoneHealth";

function App() {
  const [selectedModule, setSelectedModule] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            🦴 Bone Health AI Assessment
          </h1>
          <p className="text-gray-600 mt-1">
            Multi-ViT Enhanced CNN Framework for Bone Health Prediction
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {!selectedModule ? (
          <ModuleSelector onSelect={setSelectedModule} />
        ) : selectedModule === "fracture" ? (
          <FractureDetection onBack={() => setSelectedModule(null)} />
        ) : (
          <BoneHealth onBack={() => setSelectedModule(null)} />
        )}
      </main>

      <footer className="text-center py-6 text-gray-600 text-sm">
        Powered by Swin, DeiT, ViT Transformers with Explainable AI
      </footer>
    </div>
  );
}

export default App;
