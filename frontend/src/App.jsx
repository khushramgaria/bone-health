import { Routes, Route, Navigate } from "react-router-dom";
import HomePage from "./pages/HomePage";
import FractureDetection from "./pages/FractureDetection";
import ModelPerformance from "./pages/ModelPerformance";
import BoneDensityAnalysis from "./pages/BoneDensityAnalysis";
import ReportPage from "./pages/ReportPage";
import BoneHealthReport from "./pages/BoneHealthReport";

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/fracture-detection" element={<FractureDetection />} />
        <Route path="/model-performance" element={<ModelPerformance />} />
        <Route path="/bone-density" element={<BoneDensityAnalysis />} />
        <Route path="/report" element={<ReportPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
        <Route path="/bone-health-report" element={<BoneHealthReport />} />
      </Routes>
    </div>
  );
}

export default App;
