const ModuleSelector = ({ onSelect }) => {
  return (
    <div className="grid md:grid-cols-3 gap-8 max-w-7xl mx-auto">
      {/* Module 1: Bone Health */}
      <div
        onClick={() => onSelect("bone-health")}
        className="bg-white rounded-2xl shadow-lg p-8 cursor-pointer hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 border-2 border-transparent hover:border-blue-500"
      >
        <div className="text-6xl mb-4">🦴</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-3">
          MODULE 1: Bone Health
        </h2>
        <p className="text-gray-600 mb-4">DEXA-based BMD analysis</p>
        <button className="mt-6 w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition">
          Select Module
        </button>
      </div>

      {/* Module 2: Fracture Detection */}
      <div
        onClick={() => onSelect("fracture")}
        className="bg-white rounded-2xl shadow-lg p-8 cursor-pointer hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 border-2 border-transparent hover:border-indigo-500"
      >
        <div className="text-6xl mb-4">🩻</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-3">
          MODULE 2: Fracture Detection
        </h2>
        <p className="text-gray-600 mb-4">X-ray based fracture analysis</p>
        <button className="mt-6 w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition">
          Select Module
        </button>
      </div>

      {/* Model Performance Dashboard */}
      <div
        onClick={() => onSelect("performance")}
        className="bg-white rounded-2xl shadow-lg p-8 cursor-pointer hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 border-2 border-transparent hover:border-green-500"
      >
        <div className="text-6xl mb-4">📊</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-3">
          Model Performance
        </h2>
        <p className="text-gray-600 mb-4">View all evaluation metrics</p>
        <ul className="text-sm text-gray-700 space-y-1 mb-4">
          <li>✓ Confusion Matrix</li>
          <li>✓ ROC Curve & AUC</li>
          <li>✓ Precision/Recall/F1</li>
          <li>✓ Calibration Curve</li>
        </ul>
        <button className="mt-6 w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition">
          View Metrics
        </button>
      </div>
    </div>
  );
};

export default ModuleSelector;
