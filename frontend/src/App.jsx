import { Routes, Route, Navigate } from "react-router-dom";
import HomePage from "./pages/HomePage";
import FractureDetection from "./pages/FractureDetection";
import ModelPerformance from "./pages/ModelPerformance";
import BoneDensityAnalysis from "./pages/BoneDensityAnalysis";

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/fracture-detection" element={<FractureDetection />} />
        {/*<Route path="/model-performance" element={<ModelPerformance />} />*/}
        <Route path="/bone-density" element={<BoneDensityAnalysis />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default App;
