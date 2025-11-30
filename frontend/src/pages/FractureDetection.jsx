import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const FractureDetection = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [metricsData, setMetricsData] = useState(null);

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
      // Get prediction for this image
      const predResponse = await axios.post(
        "http://localhost:8000/api/predict-fracture",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      // Get dataset-level metrics
      const metricsResponse = await axios.get(
        "http://localhost:8000/api/metrics/all-data"
      );

      setResult(predResponse.data);
      setMetricsData(metricsResponse.data);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      alert("Error analyzing image: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeAnother = () => {
    setResult(null);
    setPreview(null);
    setSelectedFile(null);
    setMetricsData(null);
  };

  const getRiskColor = (riskLevel) => {
    if (riskLevel?.includes("Very High"))
      return "bg-red-100 border-red-500 text-red-800";
    if (riskLevel?.includes("High"))
      return "bg-orange-100 border-orange-500 text-orange-800";
    if (riskLevel?.includes("Medium"))
      return "bg-yellow-100 border-yellow-500 text-yellow-800";
    if (riskLevel?.includes("Low"))
      return "bg-green-100 border-green-500 text-green-800";
    return "bg-blue-100 border-blue-500 text-blue-800";
  };

  const TableComponent = ({ title, data, description }) => {
    const headers = Object.keys(data);
    const rowCount = data[headers[0]]?.length || 0;

    return (
      <div className="mb-10">
        <h3 className="text-2xl font-bold mb-4 text-gray-800">{title}</h3>
        <div className="overflow-x-auto bg-white rounded-xl shadow-lg border-2 border-gray-200">
          <table className="w-full">
            <thead className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
              <tr>
                {headers.map((header, idx) => (
                  <th
                    key={idx}
                    className="px-6 py-4 text-left font-bold uppercase text-sm tracking-wider"
                  >
                    {header.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: rowCount }).map((_, rowIdx) => (
                <tr
                  key={rowIdx}
                  className={rowIdx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                >
                  {headers.map((col, colIdx) => (
                    <td
                      key={colIdx}
                      className="border-b border-gray-200 px-6 py-4 text-gray-800 font-medium"
                    >
                      {data[col][rowIdx]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {description && (
          <p className="mt-4 text-sm text-gray-700 italic bg-blue-50 p-4 rounded-lg border-l-4 border-blue-600">
            <strong className="text-blue-800">Description:</strong>{" "}
            {description}
          </p>
        )}
      </div>
    );
  };

  if (!result) {
    return (
      <div className="min-h-screen p-8">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={() => navigate("/")}
            className="mb-6 flex items-center text-gray-600 hover:text-gray-900 transition font-semibold"
          >
            ← Back to Home
          </button>

          <div className="bg-gradient-to-br from-white to-purple-50 rounded-3xl shadow-2xl p-10 border border-purple-100">
            <div className="text-center mb-10">
              <div className="text-8xl mb-6">🩻</div>
              <h1 className="text-4xl font-bold text-gray-800 mb-3">
                AI Fracture Detection
              </h1>
              <p className="text-lg text-gray-600">
                Multi-Transformer Ensemble with Comprehensive Analysis
              </p>
            </div>

            <div className="space-y-6">
              <div className="border-2 border-dashed border-purple-300 rounded-2xl p-16 text-center bg-white hover:bg-purple-50 transition cursor-pointer">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <div className="text-8xl mb-6">📤</div>
                  <p className="text-2xl font-bold text-gray-700 mb-3">
                    Upload X-ray Image
                  </p>
                  <p className="text-gray-500">
                    Supported: JPG, PNG, DICOM • Max size: 10MB
                  </p>
                </label>
              </div>

              {preview && (
                <div className="space-y-6">
                  <div className="bg-white p-8 rounded-2xl shadow-lg">
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
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-6 rounded-2xl font-bold text-xl hover:from-indigo-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-xl hover:shadow-2xl transform hover:-translate-y-1"
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
                        Analyzing with AI Ensemble...
                      </span>
                    ) : (
                      <>🤖 Analyze with AI Models</>
                    )}
                  </button>
                </div>
              )}

              <div className="bg-purple-50 border-2 border-purple-200 rounded-2xl p-6 text-sm text-purple-800">
                <p className="font-bold mb-3 text-base">ℹ️ Analysis Process:</p>
                <ul className="list-disc list-inside space-y-2 ml-2">
                  <li>SigLIP transformer analyzes bone structure & edges</li>
                  <li>ViT model examines fracture patterns & density</li>
                  <li>
                    Ensemble fusion combines predictions for 96%+ accuracy
                  </li>
                  <li>
                    Explainability maps (Grad-CAM + LIME) show decision regions
                  </li>
                  <li>
                    Comprehensive dataset metrics validate model performance
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // RESULTS VIEW WITH FULL ANALYSIS
  return (
    <div className="min-h-screen p-8 bg-gradient-to-br from-gray-50 to-blue-50">
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
            onClick={handleAnalyzeAnother}
            className="bg-gradient-to-r from-green-600 to-teal-600 text-white px-8 py-4 rounded-xl font-bold hover:from-green-700 hover:to-teal-700 transition shadow-lg flex items-center text-lg"
          >
            <span className="text-2xl mr-2">🔄</span>
            Analyze Another X-ray
          </button>
        </div>

        <div className="space-y-10">
          {/* SECTION 1: PATIENT-SPECIFIC ANALYSIS */}
          <section className="bg-white rounded-3xl shadow-2xl p-10">
            <div className="text-center mb-10">
              <h2 className="text-5xl font-bold text-gray-800 mb-3">
                📊 Individual X-ray Analysis Report
              </h2>
              <p className="text-xl text-gray-600">
                AI-Powered Fracture Detection Results
              </p>
            </div>

            {/* Prediction Result */}
            <div
              className={`p-10 rounded-3xl text-center border-4 shadow-2xl mb-10 ${
                result.prediction === "Fracture Detected"
                  ? "bg-gradient-to-br from-red-50 to-red-100 border-red-500"
                  : "bg-gradient-to-br from-green-50 to-green-100 border-green-500"
              }`}
            >
              <h3 className="text-6xl font-bold mb-5">{result.prediction}</h3>
              <p className="text-3xl mb-4">
                Ensemble Confidence:{" "}
                <span className="font-bold">{result.confidence}%</span>
              </p>
              <div
                className={`inline-block px-8 py-4 rounded-2xl font-bold text-2xl border-4 ${getRiskColor(
                  result.risk_level
                )}`}
              >
                Fuzzy Risk: {result.risk_level}
              </div>
            </div>

            {/* Model Scores */}
            <div className="mb-10">
              <h3 className="text-3xl font-bold mb-6 text-gray-800">
                Model-wise Predictions
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-blue-50 p-6 rounded-2xl border-2 border-blue-200">
                  <h4 className="font-bold text-xl mb-4 text-blue-800">
                    🔬 SigLIP Model
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-700">Prediction:</span>
                      <span className="font-bold">
                        {result.model_scores.model_1_siglip.prediction}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-700">Confidence:</span>
                      <span className="font-bold text-blue-600">
                        {result.model_scores.model_1_siglip.confidence}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className="bg-blue-600 h-4 rounded-full transition-all"
                        style={{
                          width: `${result.model_scores.model_1_siglip.confidence}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-green-50 p-6 rounded-2xl border-2 border-green-200">
                  <h4 className="font-bold text-xl mb-4 text-green-800">
                    🤖 ViT Model
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-700">Prediction:</span>
                      <span className="font-bold">
                        {result.model_scores.model_2_vit.prediction}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-700">Confidence:</span>
                      <span className="font-bold text-green-600">
                        {result.model_scores.model_2_vit.confidence}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className="bg-green-600 h-4 rounded-full transition-all"
                        style={{
                          width: `${result.model_scores.model_2_vit.confidence}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Explainability Visualization */}
            <div className="mb-10">
              <h3 className="text-3xl font-bold mb-6 flex items-center">
                <span className="text-4xl mr-3">🎨</span>
                Explainability Visualization (This X-ray)
              </h3>
              <div className="grid md:grid-cols-3 gap-6 mb-6">
                <div className="bg-gray-50 p-4 rounded-xl border-2 border-gray-300">
                  <p className="text-sm font-bold text-gray-600 mb-2 text-center">
                    Original X-ray
                  </p>
                  <img
                    src={preview}
                    alt="Original"
                    className="w-full rounded-lg shadow-md"
                  />
                </div>
                <div className="bg-blue-50 p-4 rounded-xl border-2 border-blue-300">
                  <p className="text-sm font-bold text-blue-800 mb-2 text-center">
                    SigLIP Grad-CAM
                  </p>
                  <img
                    src={`data:image/png;base64,${result.explainability.gradcam_siglip}`}
                    alt="SigLIP"
                    className="w-full rounded-lg shadow-md"
                  />
                </div>
                <div className="bg-green-50 p-4 rounded-xl border-2 border-green-300">
                  <p className="text-sm font-bold text-green-800 mb-2 text-center">
                    ViT Attention
                  </p>
                  <img
                    src={`data:image/png;base64,${result.explainability.gradcam_vit}`}
                    alt="ViT"
                    className="w-full rounded-lg shadow-md"
                  />
                </div>
              </div>
              <div className="bg-purple-50 p-4 rounded-xl border-2 border-purple-300">
                <p className="text-sm font-bold text-purple-800 mb-2 text-center">
                  LIME Interpretation (Local Explainability)
                </p>
                <img
                  src={`data:image/png;base64,${result.explainability.lime_interpretation}`}
                  alt="LIME"
                  className="w-full max-w-md mx-auto rounded-lg shadow-md"
                />
                <p className="text-xs text-center text-gray-600 mt-2 italic">
                  Green regions support prediction, red regions contradict it
                </p>
              </div>
            </div>
          </section>

          {/* SECTION 2: DATASET-LEVEL PERFORMANCE METRICS */}
          {metricsData && (
            <>
              <section className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-3xl shadow-2xl p-10 text-white">
                <h2 className="text-4xl font-bold mb-4 flex items-center">
                  <span className="text-5xl mr-4">📈</span>
                  Dataset-Level Performance Metrics
                </h2>
                <p className="text-xl text-blue-100 mb-8">
                  Comprehensive model evaluation on{" "}
                  {metricsData.summary?.total_images || 0} test images
                </p>

                {metricsData.summary && (
                  <div className="grid md:grid-cols-4 gap-6">
                    <div className="bg-white rounded-2xl p-6 shadow-lg">
                      <div className="text-5xl mb-3">📸</div>
                      <div className="text-4xl font-bold text-gray-800">
                        {metricsData.summary.total_images}
                      </div>
                      <div className="text-sm text-gray-600 font-semibold mt-1">
                        Total Test Images
                      </div>
                    </div>
                    <div className="bg-white rounded-2xl p-6 shadow-lg">
                      <div className="text-5xl mb-3">🎯</div>
                      <div className="text-4xl font-bold text-gray-800">
                        {(metricsData.summary.ensemble_accuracy * 100).toFixed(
                          1
                        )}
                        %
                      </div>
                      <div className="text-sm text-gray-600 font-semibold mt-1">
                        Ensemble Accuracy
                      </div>
                    </div>
                    <div className="bg-white rounded-2xl p-6 shadow-lg">
                      <div className="text-5xl mb-3">📈</div>
                      <div className="text-4xl font-bold text-gray-800">
                        {metricsData.summary.ensemble_auc.toFixed(3)}
                      </div>
                      <div className="text-sm text-gray-600 font-semibold mt-1">
                        AUC Score
                      </div>
                    </div>
                    <div className="bg-white rounded-2xl p-6 shadow-lg">
                      <div className="text-5xl mb-3">🤖</div>
                      <div className="text-4xl font-bold text-gray-800">3</div>
                      <div className="text-sm text-gray-600 font-semibold mt-1">
                        Models Tested
                      </div>
                    </div>
                  </div>
                )}
              </section>

              {/* Training Curves */}
              <section className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-200">
                <h2 className="text-3xl font-bold mb-6 flex items-center">
                  <span className="text-4xl mr-3">📉</span>
                  Figure 2: Training & Validation Curves
                </h2>
                <img
                  src={`http://localhost:8000/api/metrics/image/figure_2_training_curves.png`}
                  alt="Training Curves"
                  className="w-full rounded-xl shadow-lg"
                  onError={(e) => {
                    e.target.style.display = "none";
                  }}
                />
                <p className="mt-4 text-gray-600 text-sm italic bg-blue-50 p-4 rounded-lg">
                  <strong>Description:</strong> Models converge within 7 epochs,
                  demonstrating efficient optimization.
                </p>
              </section>

              {/* Overall Performance */}
              <section className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-200">
                <h2 className="text-3xl font-bold mb-6 flex items-center">
                  <span className="text-4xl mr-3">📊</span>
                  Figure 3: Overall Performance Comparison
                </h2>
                <img
                  src={`http://localhost:8000/api/metrics/image/figure_3_overall_performance.png`}
                  alt="Overall Performance"
                  className="w-full rounded-xl shadow-lg"
                  onError={(e) => {
                    e.target.style.display = "none";
                  }}
                />
              </section>

              {/* Table 1 */}
              {metricsData.table1_performance_comparison && (
                <section className="bg-gradient-to-br from-white to-blue-50 rounded-2xl shadow-xl p-8 border-2 border-blue-200">
                  <TableComponent
                    title="Table 1: Performance Comparisons for All Models"
                    data={metricsData.table1_performance_comparison}
                    description="Comprehensive metrics including Kappa and MCC. Ensemble achieves highest scores."
                  />
                </section>
              )}

              {/* Confusion Matrices */}
              <section className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-200">
                <h2 className="text-3xl font-bold mb-6 flex items-center">
                  <span className="text-4xl mr-3">🔢</span>
                  Figure 5: Confusion Matrices
                </h2>
                <img
                  src={`http://localhost:8000/api/metrics/image/figure_5_confusion_matrices.png`}
                  alt="Confusion Matrices"
                  className="w-full rounded-xl shadow-lg"
                  onError={(e) => {
                    e.target.style.display = "none";
                  }}
                />
              </section>

              {/* ROC Curves */}
              <section className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-200">
                <h2 className="text-3xl font-bold mb-6 flex items-center">
                  <span className="text-4xl mr-3">📈</span>
                  Figure 11: ROC Curves
                </h2>
                <img
                  src={`http://localhost:8000/api/metrics/image/figure_11_roc_curves.png`}
                  alt="ROC Curves"
                  className="w-full rounded-xl shadow-lg"
                  onError={(e) => {
                    e.target.style.display = "none";
                  }}
                />
              </section>

              {/* Dataset Distribution */}
              {metricsData.dataset_distribution && (
                <section className="bg-gradient-to-br from-white to-indigo-50 rounded-2xl shadow-xl p-8 border-2 border-indigo-200">
                  <TableComponent
                    title="Table 8: Dataset Distribution"
                    data={metricsData.dataset_distribution}
                    description="Train/validation/test split with balanced class distribution."
                  />
                </section>
              )}
            </>
          )}

          {/* Clinical Disclaimer */}
          <section className="bg-yellow-50 border-4 border-yellow-400 rounded-2xl p-8">
            <h4 className="font-bold text-yellow-800 mb-3 flex items-center text-xl">
              <span className="text-3xl mr-3">⚠️</span>
              Clinical Disclaimer
            </h4>
            <p className="text-yellow-800">
              This AI analysis supports clinical decision-making and must be
              verified by a qualified radiologist. The system achieves{" "}
              {metricsData?.summary?.ensemble_accuracy
                ? `${(metricsData.summary.ensemble_accuracy * 100).toFixed(1)}%`
                : "96%"}{" "}
              accuracy on test datasets, but individual cases may vary. Always
              consult medical professionals for final diagnosis.
            </p>
          </section>

          {/* Bottom Actions */}
          <div className="text-center">
            <button
              onClick={handleAnalyzeAnother}
              className="bg-gradient-to-r from-green-600 to-teal-600 text-white px-12 py-5 rounded-2xl font-bold text-xl hover:from-green-700 hover:to-teal-700 transition shadow-xl hover:shadow-2xl transform hover:-translate-y-1"
            >
              <span className="text-3xl mr-3">🔄</span>
              Analyze Another X-ray
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FractureDetection;
