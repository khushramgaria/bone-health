import { useState } from "react";
import axios from "axios";

const BoneHealth = ({ onBack }) => {
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
        "http://localhost:8000/api/predict-bone-health",
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
          <span className="text-4xl mr-3">🦴</span>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">
              Bone Health Prediction
            </h2>
            <p className="text-gray-600">
              Upload DEXA scan or enter BMD values
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
                id="file-upload-dexa"
              />
              <label
                htmlFor="file-upload-dexa"
                className="cursor-pointer inline-block"
              >
                <div className="text-6xl mb-4">📤</div>
                <p className="text-lg font-semibold text-gray-700 mb-2">
                  Click to upload DEXA scan
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
                  className="w-full bg-blue-600 text-white py-4 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <span className="animate-spin mr-2">⏳</span>
                      Analyzing bone density...
                    </span>
                  ) : (
                    "Predict Bone Health"
                  )}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Category */}
            <div
              className={`p-6 rounded-xl text-center ${
                result.category === "Normal"
                  ? "bg-green-50 border-2 border-green-500"
                  : result.category === "Osteopenia"
                  ? "bg-yellow-50 border-2 border-yellow-500"
                  : "bg-red-50 border-2 border-red-500"
              }`}
            >
              <h3 className="text-3xl font-bold mb-2">{result.category}</h3>
              <p className="text-xl">Confidence: {result.confidence}%</p>
            </div>

            {/* Metrics */}
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <p className="text-gray-600 text-sm mb-1">BMD Value</p>
                <p className="text-2xl font-bold">{result.bmd_value}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <p className="text-gray-600 text-sm mb-1">T-Score</p>
                <p className="text-2xl font-bold">{result.t_score}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <p className="text-gray-600 text-sm mb-1">Z-Score</p>
                <p className="text-2xl font-bold">{result.z_score}</p>
              </div>
            </div>

            {/* Risk Score */}
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="font-semibold">
                Fracture Risk Score:{" "}
                <span className="text-xl">{result.risk_score}</span>
              </p>
            </div>

            {/* Grad-CAM */}
            <div>
              <h3 className="text-xl font-bold mb-4">
                🔍 Grad-CAM Visualization
              </h3>
              <img
                src={`data:image/png;base64,${result.gradcam}`}
                alt="Grad-CAM"
                className="max-h-96 mx-auto rounded-lg shadow-lg"
              />
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
              Analyze Another Scan
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default BoneHealth;
