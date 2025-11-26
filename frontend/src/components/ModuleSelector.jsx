const ModuleSelector = ({ onSelect }) => {
  return (
    <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
      {/* Module 1: Bone Health Prediction */}
      <div
        onClick={() => onSelect("bone-health")}
        className="bg-white rounded-2xl shadow-lg p-8 cursor-pointer hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 border-2 border-transparent hover:border-blue-500"
      >
        <div className="text-6xl mb-4">🦴</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-3">
          MODULE 1: Bone Health Prediction
        </h2>
        <p className="text-gray-600 mb-4">
          DEXA-based bone mineral density analysis
        </p>
        <ul className="text-sm text-gray-700 space-y-2">
          <li>✓ BMD value prediction</li>
          <li>✓ T-score / Z-score calculation</li>
          <li>✓ Osteoporosis risk assessment</li>
          <li>✓ Grad-CAM visualization</li>
        </ul>
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
        <p className="text-gray-600 mb-4">
          X-ray based fracture identification
        </p>
        <ul className="text-sm text-gray-700 space-y-2">
          <li>✓ Multi-model ensemble prediction</li>
          <li>✓ Confidence scoring</li>
          <li>✓ Fuzzy logic risk evaluation</li>
          <li>✓ Grad-CAM + LIME + Attention maps</li>
        </ul>
        <button className="mt-6 w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition">
          Select Module
        </button>
      </div>
    </div>
  );
};

export default ModuleSelector;
