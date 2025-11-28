import { useState } from "react";
import axios from "axios";

const FractureDetection = ({ onBack }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:8000/api/predict-fracture",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResult(response.data);
    } catch (error) {
      alert("Error analyzing image: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    if (riskLevel.includes("Very High"))
      return "bg-red-100 border-red-500 text-red-800";
    if (riskLevel.includes("High"))
      return "bg-orange-100 border-orange-500 text-orange-800";
    if (riskLevel.includes("Medium"))
      return "bg-yellow-100 border-yellow-500 text-yellow-800";
    if (riskLevel.includes("Low"))
      return "bg-green-100 border-green-500 text-green-800";
    return "bg-blue-100 border-blue-500 text-blue-800";
  };

  return (
    <div className="max-w-6xl mx-auto">
      <button
        onClick={onBack}
        className="mb-6 flex items-center text-gray-600 hover:text-gray-900"
      >
        ← Back to Modules
      </button>

      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="flex items-center mb-6">
          <span className="text-4xl mr-3">🩻</span>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">
              AI Fracture Detection Model
            </h2>
            <p className="text-gray-600">
              Multi-Transformer Ensemble System with Fuzzy Logic
            </p>
          </div>
        </div>

        {/* Upload Section */}
        {!result && (
          <div className="space-y-6">
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer inline-block"
              >
                <div className="text-6xl mb-4">📤</div>
                <p className="text-lg font-semibold text-gray-700 mb-2">
                  Click to upload X-ray image
                </p>
                <p className="text-sm text-gray-500">
                  Supported formats: JPG, PNG, DICOM
                </p>
              </label>
            </div>

            {preview && (
              <div className="space-y-4">
                <h3 className="font-semibold text-gray-800">Preview:</h3>
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-96 mx-auto rounded-lg shadow-md"
                />
                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="w-full bg-indigo-600 text-white py-4 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <span className="animate-spin mr-2">⏳</span>
                      Analyzing with 2 Transformers + Ensemble...
                    </span>
                  ) : (
                    "Analyze X-ray with AI"
                  )}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Final Prediction with Fuzzy Logic */}
            <div
              className={`p-6 rounded-xl text-center border-2 ${
                result.prediction === "Fracture Detected"
                  ? "bg-red-50 border-red-500"
                  : "bg-green-50 border-green-500"
              }`}
            >
              <h3 className="text-3xl font-bold mb-2">{result.prediction}</h3>
              <p className="text-xl mb-2">
                Ensemble Confidence: {result.confidence}%
              </p>
              <div
                className={`inline-block px-4 py-2 rounded-lg font-semibold ${getRiskColor(
                  result.risk_level
                )}`}
              >
                Fuzzy Risk Level: {result.risk_level}
              </div>
            </div>

            {/* Individual Model Scores */}
            <div className="bg-gray-50 p-6 rounded-lg">
              <h3 className="font-bold text-lg mb-4">
                🤖 Individual Model Predictions
              </h3>
              <div className="space-y-3">
                {/* SigLIP Model */}
                <div className="bg-white p-4 rounded-lg border-2 border-blue-200">
                  <div className="flex justify-between items-center mb-2">
                    <div>
                      <span className="font-semibold text-lg">
                        Model 1:{" "}
                        {result.model_scores.model_1_siglip.transformer}
                      </span>
                      <p className="text-sm text-gray-600">
                        {result.model_scores.model_1_siglip.prediction}
                      </p>
                    </div>
                    <div className="text-right">
                      <span className="text-2xl font-bold text-blue-600">
                        {result.model_scores.model_1_siglip.confidence}%
                      </span>
                      <p className="text-xs text-gray-500">
                        Fuzzy: {result.model_scores.model_1_siglip.fuzzy_risk}
                      </p>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{
                        width: `${result.model_scores.model_1_siglip.confidence}%`,
                      }}
                    ></div>
                  </div>
                </div>

                {/* ViT Model */}
                <div className="bg-white p-4 rounded-lg border-2 border-green-200">
                  <div className="flex justify-between items-center mb-2">
                    <div>
                      <span className="font-semibold text-lg">
                        Model 2: {result.model_scores.model_2_vit.transformer}
                      </span>
                      <p className="text-sm text-gray-600">
                        {result.model_scores.model_2_vit.prediction}
                      </p>
                    </div>
                    <div className="text-right">
                      <span className="text-2xl font-bold text-green-600">
                        {result.model_scores.model_2_vit.confidence}%
                      </span>
                      <p className="text-xs text-gray-500">
                        Fuzzy: {result.model_scores.model_2_vit.fuzzy_risk}
                      </p>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-600 h-2 rounded-full"
                      style={{
                        width: `${result.model_scores.model_2_vit.confidence}%`,
                      }}
                    ></div>
                  </div>
                  {result.model_scores.model_2_vit.observation && (
                    <p className="text-xs text-gray-600 mt-2 italic">
                      {result.model_scores.model_2_vit.observation}
                    </p>
                  )}
                </div>

                {/* Ensemble Result */}
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-4 rounded-lg border-2 border-indigo-300">
                  <div className="flex justify-between items-center mb-2">
                    <div>
                      <span className="font-bold text-lg">
                        Ensemble (Average Fusion)
                      </span>
                      <p className="text-sm text-gray-600">
                        Combined prediction from both models
                      </p>
                    </div>
                    <div className="text-right">
                      <span className="text-3xl font-bold text-indigo-700">
                        {result.model_scores.ensemble.confidence}%
                      </span>
                      <p className="text-xs text-gray-500">
                        Fuzzy: {result.model_scores.ensemble.fuzzy_risk}
                      </p>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-indigo-600 to-purple-600 h-3 rounded-full"
                      style={{
                        width: `${result.model_scores.ensemble.confidence}%`,
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Explainability Visualizations */}
            <div>
              <h3 className="text-xl font-bold mb-4">
                🔍 Explainability Visualizations (Grad-CAM & LIME)
              </h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">SigLIP Grad-CAM</h4>
                  <p className="text-xs text-gray-600 mb-2">
                    Model 1 attention regions
                  </p>
                  <img
                    src={`data:image/png;base64,${result.explainability.gradcam_siglip}`}
                    alt="SigLIP Grad-CAM"
                    className="w-full rounded-lg shadow"
                  />
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">ViT Attention Map</h4>
                  <p className="text-xs text-gray-600 mb-2">
                    Model 2 attention regions
                  </p>
                  <img
                    src={`data:image/png;base64,${result.explainability.gradcam_vit}`}
                    alt="ViT Attention"
                    className="w-full rounded-lg shadow"
                  />
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">LIME Interpretation</h4>
                  <p className="text-xs text-gray-600 mb-2">
                    Feature importance map
                  </p>
                  <img
                    src={`data:image/png;base64,${result.explainability.lime_interpretation}`}
                    alt="LIME"
                    className="w-full rounded-lg shadow"
                  />
                </div>
              </div>
            </div>

            {/* Fuzzy Logic Explanation */}
            <div className="bg-purple-50 border border-purple-300 rounded-lg p-4">
              <h4 className="font-semibold mb-2">
                ℹ️ Fuzzy Logic Risk Assessment
              </h4>
              <p className="text-sm text-gray-700 mb-2">
                Our system uses 5-level fuzzy logic to translate numeric
                predictions into clinical risk categories:
              </p>
              <div className="grid grid-cols-5 gap-2 text-xs text-center">
                <div className="bg-blue-100 p-2 rounded">
                  Very Low
                  <br />
                  (0-20%)
                </div>
                <div className="bg-green-100 p-2 rounded">
                  Low
                  <br />
                  (20-40%)
                </div>
                <div className="bg-yellow-100 p-2 rounded">
                  Medium
                  <br />
                  (40-60%)
                </div>
                <div className="bg-orange-100 p-2 rounded">
                  High
                  <br />
                  (60-80%)
                </div>
                <div className="bg-red-100 p-2 rounded">
                  Very High
                  <br />
                  (80-100%)
                </div>
              </div>
            </div>

            {/* Safety Message */}
            <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4">
              <p className="text-sm text-yellow-800">
                ⚠️ This AI prediction supports clinical decision-making but does
                not replace professional radiologist judgment.
              </p>
            </div>

            {/* Action Button */}
            <button
              onClick={() => {
                setResult(null);
                setPreview(null);
                setSelectedFile(null);
              }}
              className="w-full bg-gray-600 text-white py-3 rounded-lg font-semibold hover:bg-gray-700 transition"
            >
              Analyze Another X-ray
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FractureDetection;
