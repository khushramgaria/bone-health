import { useState } from "react";
import axios from "axios";

const BoneHealth = ({ onBack }) => {
  const [mode, setMode] = useState(null); // 'image' or 'values'
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // Form values
  const [formData, setFormData] = useState({
    bmd_value: "",
    t_score: "",
    z_score: "",
    age: "",
    gender: "",
  });

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleAnalyzeImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formDataObj = new FormData();
    formDataObj.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:8000/api/predict-bone-health/image",
        formDataObj,
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

  const handleAnalyzeValues = async () => {
    setLoading(true);
    const formDataObj = new FormData();

    if (formData.bmd_value) formDataObj.append("bmd_value", formData.bmd_value);
    if (formData.t_score) formDataObj.append("t_score", formData.t_score);
    if (formData.z_score) formDataObj.append("z_score", formData.z_score);
    if (formData.age) formDataObj.append("age", formData.age);
    if (formData.gender) formDataObj.append("gender", formData.gender);

    try {
      const response = await axios.post(
        "http://localhost:8000/api/predict-bone-health/values",
        formDataObj
      );
      setResult(response.data);
    } catch (error) {
      alert("Error analyzing values: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Mode selection screen
  if (!mode) {
    return (
      <div className="max-w-6xl mx-auto">
        <button
          onClick={onBack}
          className="mb-6 flex items-center text-gray-600 hover:text-gray-900"
        >
          ← Back to Modules
        </button>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="text-center mb-8">
            <span className="text-6xl mb-4 block">🦴</span>
            <h2 className="text-3xl font-bold text-gray-800 mb-2">
              Bone Health Prediction
            </h2>
            <p className="text-gray-600">Choose input method</p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Upload DEXA Scan */}
            <div
              onClick={() => setMode("image")}
              className="bg-gradient-to-br from-blue-50 to-indigo-50 p-8 rounded-xl border-2 border-blue-200 cursor-pointer hover:shadow-lg transition hover:-translate-y-1"
            >
              <div className="text-5xl mb-4">📤</div>
              <h3 className="text-xl font-bold text-gray-800 mb-2">
                Upload DEXA Scan
              </h3>
              <p className="text-gray-600 mb-4">
                Upload your DEXA scan image for AI-powered analysis
              </p>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>✓ AI analyzes bone density</li>
                <li>✓ Automatic BMD calculation</li>
                <li>✓ Visual heatmap explanation</li>
              </ul>
            </div>

            {/* Enter BMD Values */}
            <div
              onClick={() => setMode("values")}
              className="bg-gradient-to-br from-green-50 to-teal-50 p-8 rounded-xl border-2 border-green-200 cursor-pointer hover:shadow-lg transition hover:-translate-y-1"
            >
              <div className="text-5xl mb-4">📝</div>
              <h3 className="text-xl font-bold text-gray-800 mb-2">
                Enter BMD Values
              </h3>
              <p className="text-gray-600 mb-4">
                Input your known BMD values for instant classification
              </p>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>✓ Manual T-score input</li>
                <li>✓ BMD value analysis</li>
                <li>✓ Instant classification</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Image upload mode
  if (mode === "image" && !result) {
    return (
      <div className="max-w-6xl mx-auto">
        <button
          onClick={() => setMode(null)}
          className="mb-6 flex items-center text-gray-600 hover:text-gray-900"
        >
          ← Back to Mode Selection
        </button>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="flex items-center mb-6">
            <span className="text-4xl mr-3">📤</span>
            <div>
              <h2 className="text-2xl font-bold text-gray-800">
                Upload DEXA Scan
              </h2>
              <p className="text-gray-600">
                Upload your bone density scan image
              </p>
            </div>
          </div>

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
                <div className="text-6xl mb-4">🦴</div>
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
                  onClick={handleAnalyzeImage}
                  disabled={loading}
                  className="w-full bg-blue-600 text-white py-4 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <span className="animate-spin mr-2">⏳</span>
                      Analyzing bone density with Gemini AI...
                    </span>
                  ) : (
                    "Predict Bone Health"
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Manual values mode
  if (mode === "values" && !result) {
    return (
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => setMode(null)}
          className="mb-6 flex items-center text-gray-600 hover:text-gray-900"
        >
          ← Back to Mode Selection
        </button>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="flex items-center mb-6">
            <span className="text-4xl mr-3">📝</span>
            <div>
              <h2 className="text-2xl font-bold text-gray-800">
                Enter BMD Values
              </h2>
              <p className="text-gray-600">
                Input your bone mineral density measurements
              </p>
            </div>
          </div>

          <form className="space-y-6">
            {/* BMD Value */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                BMD Value (g/cm²)
              </label>
              <input
                type="number"
                step="0.01"
                name="bmd_value"
                value={formData.bmd_value}
                onChange={handleInputChange}
                placeholder="e.g., 0.85"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* T-Score */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                T-Score
              </label>
              <input
                type="number"
                step="0.1"
                name="t_score"
                value={formData.t_score}
                onChange={handleInputChange}
                placeholder="e.g., -1.5"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                Normal: {">"} -1.0 | Osteopenia: -1.0 to -2.5 | Osteoporosis:{" "}
                {"<"} -2.5
              </p>
            </div>

            {/* Z-Score */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Z-Score (optional)
              </label>
              <input
                type="number"
                step="0.1"
                name="z_score"
                value={formData.z_score}
                onChange={handleInputChange}
                placeholder="e.g., -0.8"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Age & Gender */}
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Age (optional)
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  placeholder="e.g., 65"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Gender (optional)
                </label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
            </div>

            <button
              type="button"
              onClick={handleAnalyzeValues}
              disabled={loading || (!formData.bmd_value && !formData.t_score)}
              className="w-full bg-green-600 text-white py-4 rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <span className="animate-spin mr-2">⏳</span>
                  Analyzing values...
                </span>
              ) : (
                "Analyze Bone Health"
              )}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Results display
  if (result) {
    return (
      <div className="max-w-6xl mx-auto">
        <button
          onClick={() => {
            setResult(null);
            setPreview(null);
            setSelectedFile(null);
            setFormData({
              bmd_value: "",
              t_score: "",
              z_score: "",
              age: "",
              gender: "",
            });
          }}
          className="mb-6 flex items-center text-gray-600 hover:text-gray-900"
        >
          ← Analyze Another
        </button>

        <div className="bg-white rounded-2xl shadow-lg p-8">
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
              <p className="mt-2 text-sm text-gray-600">
                Source: {result.source}
              </p>
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

            {/* Risk Level */}
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="font-semibold">
                Fracture Risk:{" "}
                <span className="text-xl text-blue-600">
                  {result.risk_level}
                </span>
              </p>
            </div>

            {/* Clinical Interpretation */}
            <div className="bg-purple-50 p-6 rounded-lg border border-purple-200">
              <h3 className="font-bold text-lg mb-2">
                🩺 Clinical Interpretation
              </h3>
              <p className="text-gray-700">{result.interpretation}</p>
            </div>

            {/* Grad-CAM (if from image) */}
            {result.gradcam && (
              <div>
                <h3 className="text-xl font-bold mb-4">🔍 Visual Analysis</h3>
                <img
                  src={`data:image/png;base64,${result.gradcam}`}
                  alt="Grad-CAM"
                  className="max-h-96 mx-auto rounded-lg shadow-lg"
                />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default BoneHealth;
