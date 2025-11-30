import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const BoneDensityAnalysis = () => {
  const navigate = useNavigate();
  const [mode, setMode] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
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
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(response.data);
      window.scrollTo({ top: 0, behavior: "smooth" });
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
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      alert("Error analyzing values: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setPreview(null);
    setSelectedFile(null);
    setMode(null);
    setFormData({
      bmd_value: "",
      t_score: "",
      z_score: "",
      age: "",
      gender: "",
    });
  };

  const getCategoryColor = (category) => {
    if (category === "Normal") return "from-green-500 to-emerald-600";
    if (category === "Osteopenia") return "from-yellow-500 to-orange-600";
    if (category === "Osteoporosis") return "from-red-500 to-rose-600";
    return "from-blue-500 to-indigo-600";
  };

  const getCategoryBorder = (category) => {
    if (category === "Normal") return "border-green-500";
    if (category === "Osteopenia") return "border-yellow-500";
    if (category === "Osteoporosis") return "border-red-500";
    return "border-blue-500";
  };

  const getRiskBadge = (risk) => {
    if (risk === "Low" || risk === "Low Risk")
      return "bg-green-100 border-green-500 text-green-800";
    if (risk === "Medium" || risk === "Medium Risk")
      return "bg-yellow-100 border-yellow-500 text-yellow-800";
    if (risk === "High" || risk === "High Risk")
      return "bg-red-100 border-red-500 text-red-800";
    return "bg-blue-100 border-blue-500 text-blue-800";
  };

  // MODE SELECTION SCREEN
  if (!mode) {
    return (
      <div className="min-h-screen p-8 bg-gradient-to-br from-teal-50 to-cyan-50">
        <div className="max-w-6xl mx-auto">
          <button
            onClick={() => navigate("/")}
            className="mb-6 flex items-center text-gray-600 hover:text-gray-900 transition font-semibold"
          >
            ← Back to Home
          </button>

          <div className="bg-gradient-to-br from-white to-teal-50 rounded-3xl shadow-2xl p-10 border border-teal-100">
            <div className="text-center mb-10">
              <div className="text-8xl mb-6">🦴</div>
              <h1 className="text-4xl font-bold text-gray-800 mb-3">
                Bone Density Analysis
              </h1>
              <p className="text-lg text-gray-600">
                AI-Powered BMD Assessment using Vision Transformers
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              {/* Upload DEXA Scan */}
              <div
                onClick={() => setMode("image")}
                className="group cursor-pointer"
              >
                <div className="bg-gradient-to-br from-blue-50 to-indigo-100 p-8 rounded-2xl border-2 border-blue-300 hover:shadow-2xl transition-all hover:-translate-y-2 h-full">
                  <div className="text-6xl mb-4">📤</div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-3">
                    Upload DEXA Scan
                  </h3>
                  <p className="text-gray-700 mb-4">
                    Upload your DEXA scan image for AI-powered analysis
                  </p>
                  <ul className="text-sm text-gray-700 space-y-2">
                    <li className="flex items-center">
                      <span className="text-green-600 mr-2">✓</span>
                      Vision Transformer AI analyzes bone density
                    </li>
                    <li className="flex items-center">
                      <span className="text-green-600 mr-2">✓</span>
                      Automatic BMD, T-score, Z-score calculation
                    </li>
                    <li className="flex items-center">
                      <span className="text-green-600 mr-2">✓</span>
                      Visual heatmap explanation
                    </li>
                  </ul>
                </div>
              </div>

              {/* Enter BMD Values */}
              <div
                onClick={() => setMode("values")}
                className="group cursor-pointer"
              >
                <div className="bg-gradient-to-br from-green-50 to-teal-100 p-8 rounded-2xl border-2 border-green-300 hover:shadow-2xl transition-all hover:-translate-y-2 h-full">
                  <div className="text-6xl mb-4">📝</div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-3">
                    Enter BMD Values
                  </h3>
                  <p className="text-gray-700 mb-4">
                    Input your known BMD values for instant classification
                  </p>
                  <ul className="text-sm text-gray-700 space-y-2">
                    <li className="flex items-center">
                      <span className="text-green-600 mr-2">✓</span>
                      Manual T-score input
                    </li>
                    <li className="flex items-center">
                      <span className="text-green-600 mr-2">✓</span>
                      BMD value analysis
                    </li>
                    <li className="flex items-center">
                      <span className="text-green-600 mr-2">✓</span>
                      Instant WHO classification
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="mt-8 bg-teal-50 border-2 border-teal-200 rounded-2xl p-6 text-sm text-teal-800">
              <p className="font-bold mb-3 text-base">
                ℹ️ WHO Classification Standards:
              </p>
              <div className="grid md:grid-cols-3 gap-4 text-xs">
                <div className="bg-white p-3 rounded-lg">
                  <p className="font-bold text-green-700">Normal</p>
                  <p className="text-gray-600">T-score ≥ -1.0</p>
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <p className="font-bold text-yellow-700">Osteopenia</p>
                  <p className="text-gray-600">T-score -1.0 to -2.5</p>
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <p className="font-bold text-red-700">Osteoporosis</p>
                  <p className="text-gray-600">T-score ≤ -2.5</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // IMAGE UPLOAD MODE
  if (mode === "image" && !result) {
    return (
      <div className="min-h-screen p-8 bg-gradient-to-br from-teal-50 to-cyan-50">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={() => setMode(null)}
            className="mb-6 flex items-center text-gray-600 hover:text-gray-900 transition font-semibold"
          >
            ← Back to Mode Selection
          </button>

          <div className="bg-white rounded-3xl shadow-2xl p-10">
            <div className="flex items-center mb-8">
              <span className="text-5xl mr-4">📤</span>
              <div>
                <h2 className="text-3xl font-bold text-gray-800">
                  Upload DEXA Scan
                </h2>
                <p className="text-gray-600">
                  AI Vision Model will analyze your bone density scan
                </p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="border-2 border-dashed border-teal-300 rounded-2xl p-16 text-center bg-teal-50 hover:bg-teal-100 transition cursor-pointer">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload-dexa"
                />
                <label htmlFor="file-upload-dexa" className="cursor-pointer">
                  <div className="text-7xl mb-6">🦴</div>
                  <p className="text-2xl font-bold text-gray-700 mb-3">
                    Click to upload DEXA scan
                  </p>
                  <p className="text-gray-500">
                    Supported: JPG, PNG, DICOM • Max size: 10MB
                  </p>
                </label>
              </div>

              {preview && (
                <div className="space-y-6">
                  <div className="bg-gray-50 p-6 rounded-2xl">
                    <h3 className="font-bold text-xl text-gray-800 mb-4 flex items-center">
                      <span className="text-3xl mr-3">🔍</span>
                      Preview
                    </h3>
                    <img
                      src={preview}
                      alt="Preview"
                      className="max-h-96 mx-auto rounded-xl shadow-xl border-4 border-gray-200"
                    />
                  </div>

                  <button
                    onClick={handleAnalyzeImage}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-teal-600 to-cyan-600 text-white py-6 rounded-2xl font-bold text-xl hover:from-teal-700 hover:to-cyan-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-xl hover:shadow-2xl transform hover:-translate-y-1"
                  >
                    {loading ? (
                      <span className="flex items-center justify-center">
                        <svg
                          className="animate-spin h-7 w-7 mr-3"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          />
                        </svg>
                        Analyzing with AI Vision Model...
                      </span>
                    ) : (
                      <>🦴 Analyze Bone Density with AI</>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // MANUAL VALUES MODE
  if (mode === "values" && !result) {
    return (
      <div className="min-h-screen p-8 bg-gradient-to-br from-green-50 to-teal-50">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={() => setMode(null)}
            className="mb-6 flex items-center text-gray-600 hover:text-gray-900 transition font-semibold"
          >
            ← Back to Mode Selection
          </button>

          <div className="bg-white rounded-3xl shadow-2xl p-10">
            <div className="flex items-center mb-8">
              <span className="text-5xl mr-4">📝</span>
              <div>
                <h2 className="text-3xl font-bold text-gray-800">
                  Enter BMD Values
                </h2>
                <p className="text-gray-600">
                  Input your bone mineral density measurements
                </p>
              </div>
            </div>

            <form className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                {/* BMD Value */}
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">
                    BMD Value (g/cm²) <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    name="bmd_value"
                    value={formData.bmd_value}
                    onChange={handleInputChange}
                    placeholder="e.g., 0.85"
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                  />
                </div>

                {/* T-Score */}
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">
                    T-Score <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    name="t_score"
                    value={formData.t_score}
                    onChange={handleInputChange}
                    placeholder="e.g., -1.5"
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Normal: &gt; -1.0 | Osteopenia: -1.0 to -2.5 | Osteoporosis:
                    &lt; -2.5
                  </p>
                </div>
              </div>

              {/* Z-Score */}
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Z-Score (optional)
                </label>
                <input
                  type="number"
                  step="0.1"
                  name="z_score"
                  value={formData.z_score}
                  onChange={handleInputChange}
                  placeholder="e.g., -0.8"
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                />
              </div>

              {/* Age & Gender */}
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">
                    Age (optional)
                  </label>
                  <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleInputChange}
                    placeholder="e.g., 65"
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">
                    Gender (optional)
                  </label>
                  <select
                    name="gender"
                    value={formData.gender}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                  >
                    <option value="">Select gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
              </div>

              <button
                type="button"
                onClick={handleAnalyzeValues}
                disabled={loading || (!formData.bmd_value && !formData.t_score)}
                className="w-full bg-gradient-to-r from-green-600 to-teal-600 text-white py-6 rounded-2xl font-bold text-xl hover:from-green-700 hover:to-teal-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-xl hover:shadow-2xl transform hover:-translate-y-1"
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg
                      className="animate-spin h-7 w-7 mr-3"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Analyzing...
                  </span>
                ) : (
                  <>📊 Analyze Bone Health</>
                )}
              </button>

              <p className="text-xs text-gray-500 text-center">
                * At least BMD Value or T-Score is required
              </p>
            </form>
          </div>
        </div>
      </div>
    );
  }

  // RESULTS DISPLAY
  if (result) {
    return (
      <div className="min-h-screen p-8 bg-gradient-to-br from-gray-50 to-teal-50">
        <div className="max-w-7xl mx-auto">
          {/* Top Navigation */}
          <div className="sticky top-4 z-10 bg-white shadow-xl rounded-2xl p-5 mb-8 flex justify-between items-center">
            <button
              onClick={() => navigate("/")}
              className="text-gray-600 hover:text-gray-900 transition font-semibold"
            >
              ← Back to Home
            </button>
            <button
              onClick={handleReset}
              className="bg-gradient-to-r from-green-600 to-teal-600 text-white px-8 py-4 rounded-xl font-bold hover:from-green-700 hover:to-teal-700 transition shadow-lg flex items-center text-lg"
            >
              <span className="text-2xl mr-2">🔄</span>
              Analyze Another
            </button>
          </div>

          <div className="space-y-10">
            {/* Main Result Card */}
            <section className="bg-white rounded-3xl shadow-2xl p-10">
              <div className="text-center mb-10">
                <h2 className="text-5xl font-bold text-gray-800 mb-3">
                  🦴 Bone Density Analysis Report
                </h2>
                <p className="text-xl text-gray-600">
                  AI-Powered BMD Assessment
                </p>
              </div>

              {/* Category Result */}
              <div
                className={`p-10 rounded-3xl text-center border-4 shadow-2xl mb-10 bg-gradient-to-br ${getCategoryColor(
                  result.category
                )} ${getCategoryBorder(result.category)}`}
              >
                <h3 className="text-6xl font-bold mb-5 text-white">
                  {result.category}
                </h3>
                <p className="text-3xl mb-4 text-white">
                  Confidence:{" "}
                  <span className="font-bold">{result.confidence}%</span>
                </p>
                <div
                  className={`inline-block px-8 py-4 rounded-2xl font-bold text-2xl border-4 ${getRiskBadge(
                    result.risk_level
                  )}`}
                >
                  Fracture Risk: {result.risk_level}
                </div>
              </div>

              {/* Key Metrics */}
              <div className="mb-10">
                <h3 className="text-3xl font-bold mb-6 text-gray-800">
                  Key Measurements
                </h3>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-2xl border-2 border-blue-200">
                    <div className="text-4xl mb-3">📊</div>
                    <div className="text-3xl font-bold text-blue-800">
                      {result.bmd_value}
                    </div>
                    <div className="text-sm text-gray-700 font-semibold mt-1">
                      BMD (g/cm²)
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-2xl border-2 border-purple-200">
                    <div className="text-4xl mb-3">🎯</div>
                    <div className="text-3xl font-bold text-purple-800">
                      {result.t_score}
                    </div>
                    <div className="text-sm text-gray-700 font-semibold mt-1">
                      T-Score
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-2xl border-2 border-green-200">
                    <div className="text-4xl mb-3">📈</div>
                    <div className="text-3xl font-bold text-green-800">
                      {result.z_score}
                    </div>
                    <div className="text-sm text-gray-700 font-semibold mt-1">
                      Z-Score
                    </div>
                  </div>
                </div>
              </div>

              {/* Grad-CAM Visualization (only for image mode) */}
              {result.gradcam && preview && (
                <div className="mb-10">
                  <h3 className="text-3xl font-bold mb-6 flex items-center">
                    <span className="text-4xl mr-3">🎨</span>
                    AI Visual Analysis
                  </h3>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-gray-50 p-4 rounded-xl border-2 border-gray-300">
                      <p className="text-sm font-bold text-gray-600 mb-2 text-center">
                        Original DEXA Scan
                      </p>
                      <img
                        src={preview}
                        alt="Original"
                        className="w-full rounded-lg shadow-md"
                      />
                    </div>
                    <div className="bg-teal-50 p-4 rounded-xl border-2 border-teal-300">
                      <p className="text-sm font-bold text-teal-800 mb-2 text-center">
                        AI Density Analysis Heatmap
                      </p>
                      <img
                        src={`data:image/png;base64,${result.gradcam}`}
                        alt="Heatmap"
                        className="w-full rounded-lg shadow-md"
                      />
                      <p className="text-xs text-center text-gray-600 mt-2 italic">
                        AI-highlighted regions showing bone density assessment
                        areas
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Clinical Interpretation */}
              <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border-2 border-purple-300 rounded-2xl p-8">
                <h3 className="text-2xl font-bold text-purple-800 mb-4 flex items-center">
                  <span className="text-3xl mr-3">🩺</span>
                  Clinical Interpretation
                </h3>
                <p className="text-gray-800 text-lg leading-relaxed">
                  {result.interpretation}
                </p>
              </div>

              {/* WHO Classification Table */}
              <div className="mt-10">
                <h3 className="text-3xl font-bold mb-6 text-gray-800">
                  WHO Classification Reference
                </h3>
                <div className="overflow-x-auto bg-white rounded-xl shadow-lg border-2 border-gray-200">
                  <table className="w-full">
                    <thead className="bg-gradient-to-r from-teal-600 to-cyan-600 text-white">
                      <tr>
                        <th className="px-6 py-4 text-left font-bold">
                          Category
                        </th>
                        <th className="px-6 py-4 text-left font-bold">
                          T-Score Range
                        </th>
                        <th className="px-6 py-4 text-left font-bold">
                          Description
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr
                        className={
                          result.category === "Normal"
                            ? "bg-green-50 border-l-4 border-green-500"
                            : "bg-white"
                        }
                      >
                        <td className="px-6 py-4 font-bold text-green-700">
                          Normal
                        </td>
                        <td className="px-6 py-4">-1.0 or higher</td>
                        <td className="px-6 py-4">
                          Bone density is within 1 SD of young adult mean
                        </td>
                      </tr>
                      <tr
                        className={
                          result.category === "Osteopenia"
                            ? "bg-yellow-50 border-l-4 border-yellow-500"
                            : "bg-gray-50"
                        }
                      >
                        <td className="px-6 py-4 font-bold text-yellow-700">
                          Osteopenia
                        </td>
                        <td className="px-6 py-4">-1.0 to -2.5</td>
                        <td className="px-6 py-4">
                          Low bone mass, increased fracture risk
                        </td>
                      </tr>
                      <tr
                        className={
                          result.category === "Osteoporosis"
                            ? "bg-red-50 border-l-4 border-red-500"
                            : "bg-white"
                        }
                      >
                        <td className="px-6 py-4 font-bold text-red-700">
                          Osteoporosis
                        </td>
                        <td className="px-6 py-4">-2.5 or lower</td>
                        <td className="px-6 py-4">
                          Significantly increased fracture risk
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* Medical Disclaimer */}
            <section className="bg-yellow-50 border-4 border-yellow-400 rounded-2xl p-8">
              <h4 className="font-bold text-yellow-800 mb-3 flex items-center text-xl">
                <span className="text-3xl mr-3">⚠️</span>
                Medical Disclaimer
              </h4>
              <p className="text-yellow-800">
                This AI-powered bone density analysis is for screening purposes
                only and should not replace professional medical diagnosis.
                Results should be reviewed by a qualified physician or
                radiologist. Always consult healthcare professionals for
                accurate diagnosis and treatment plans.
              </p>
            </section>

            {/* Bottom Actions */}
            <div className="text-center pb-8">
              <button
                onClick={handleReset}
                className="bg-gradient-to-r from-teal-600 to-cyan-600 text-white px-12 py-5 rounded-2xl font-bold text-xl hover:from-teal-700 hover:to-cyan-700 transition shadow-xl hover:shadow-2xl transform hover:-translate-y-1"
              >
                <span className="text-3xl mr-3">🔄</span>
                Analyze Another Scan
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default BoneDensityAnalysis;
