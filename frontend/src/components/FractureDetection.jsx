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
              Upload X-ray image for analysis using 2 Medical Transformers
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
                      Running 2 Medical Transformers...
                    </span>
                  ) : (
                    "Analyze X-ray"
                  )}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Final Prediction */}
            <div
              className={`p-6 rounded-xl text-center ${
                result.prediction === "Fracture Detected"
                  ? "bg-red-50 border-2 border-red-500"
                  : "bg-green-50 border-2 border-green-500"
              }`}
            >
              <h3 className="text-3xl font-bold mb-2">{result.prediction}</h3>
              <p className="text-xl">
                Ensemble Confidence: {result.confidence}%
              </p>
              <p className="mt-2 text-lg font-semibold">
                Risk Level: {result.risk_level}
              </p>
            </div>

            {/* Model Scores Breakdown */}
            <div className="bg-gray-50 p-6 rounded-lg">
              <h3 className="font-bold text-lg mb-4">
                🤖 Transformer Model Breakdown
              </h3>
              <div className="space-y-3">
                {/* Model 1 */}
                <div className="bg-white p-4 rounded-lg border">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold">
                      Model 1: {result.model_scores.model_1_siglip.transformer}
                    </span>
                    <span className="text-2xl font-bold text-indigo-600">
                      {result.model_scores.model_1_siglip.confidence}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">
                    Prediction: {result.model_scores.model_1_siglip.prediction}
                  </p>
                </div>

                {/* Model 2 */}
                <div className="bg-white p-4 rounded-lg border">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold">
                      Model 2: {result.model_scores.model_2_vit.transformer}
                    </span>
                    <span className="text-2xl font-bold text-blue-600">
                      {result.model_scores.model_2_vit.confidence}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">
                    Prediction: {result.model_scores.model_2_vit.prediction}
                  </p>
                </div>

                {/* Ensemble */}
                <div className="bg-gradient-to-r from-indigo-50 to-blue-50 p-4 rounded-lg border-2 border-indigo-300">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-lg">
                      Final Ensemble (Average):
                    </span>
                    <span className="text-3xl font-bold text-indigo-700">
                      {result.model_scores.ensemble}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Explainability Section */}
            <div>
              <h3 className="text-xl font-bold mb-4">
                🔍 Explainability Visualizations
              </h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Grad-CAM: SigLIP</h4>
                  <p className="text-xs text-gray-600 mb-2">
                    Model 1 attention regions
                  </p>
                  <img
                    src={`data:image/png;base64,${result.explainability.gradcam_siglip}`}
                    alt="Grad-CAM SigLIP"
                    className="w-full rounded-lg shadow"
                  />
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Grad-CAM: ViT</h4>
                  <p className="text-xs text-gray-600 mb-2">
                    Model 2 attention regions
                  </p>
                  <img
                    src={`data:image/png;base64,${result.explainability.gradcam_vit}`}
                    alt="Grad-CAM ViT"
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

            {/* Safety Message */}
            <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4">
              <p className="text-sm text-yellow-800">
                ⚠️ This prediction supports clinical decision-making but does
                not replace radiologist judgement.
              </p>
            </div>

            {/* Buttons */}
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
