import { useState, useEffect } from "react";
import axios from "axios";

const ModelPerformance = ({ onBack }) => {
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(
        "http://localhost:8000/api/metrics/all-data"
      );
      setMetricsData(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching metrics:", error);
      setError(error.message);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="text-center py-20">
          <div className="text-6xl mb-4">⏳</div>
          <p className="text-xl text-gray-600">
            Loading performance metrics...
          </p>
        </div>
      </div>
    );
  }

  if (error || !metricsData) {
    return (
      <div className="max-w-7xl mx-auto">
        <button
          onClick={onBack}
          className="mb-6 flex items-center text-gray-600 hover:text-gray-900"
        >
          ← Back to Modules
        </button>
        <div className="bg-yellow-50 border-2 border-yellow-300 rounded-xl p-8 text-center">
          <div className="text-6xl mb-4">⚠️</div>
          <h3 className="text-2xl font-bold text-gray-800 mb-4">
            No Performance Metrics Available
          </h3>
          <p className="text-gray-700 mb-4">
            Please run the model evaluation script first:
          </p>
          <div className="bg-gray-800 text-green-400 px-4 py-2 rounded mb-4 font-mono text-sm">
            cd backend && python3 evaluate_model.py
          </div>
          <p className="text-sm text-gray-600">
            This will generate all performance metrics and visualizations.
          </p>
          {error && <p className="mt-4 text-red-600 text-sm">Error: {error}</p>}
        </div>
      </div>
    );
  }

  const {
    summary,
    performance_comparison,
    classwise_performance,
    dataset_distribution,
  } = metricsData;

  return (
    <div className="max-w-7xl mx-auto">
      <button
        onClick={onBack}
        className="mb-6 flex items-center text-gray-600 hover:text-gray-900"
      >
        ← Back to Modules
      </button>

      <div className="bg-white rounded-2xl shadow-lg p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-2">
            📊 Model Performance Dashboard
          </h2>
          <p className="text-gray-600">
            Multi-Transformer Ensemble Evaluation - {summary.total_images} Test
            Images
          </p>
        </div>

        {/* Models Used */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg mb-8">
          <h3 className="text-xl font-bold mb-4">🤖 Models Used</h3>
          <div className="grid md:grid-cols-3 gap-4">
            {summary.models_used.map((model, idx) => (
              <div key={idx} className="bg-white p-4 rounded-lg shadow">
                <p className="font-semibold text-gray-800">{model}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Overall Accuracy Cards */}
        <div className="grid grid-cols-3 gap-6 mb-8">
          <div className="bg-blue-50 p-6 rounded-lg border-2 border-blue-200 text-center">
            <p className="text-sm text-gray-600 mb-2">SigLIP Accuracy</p>
            <p className="text-4xl font-bold text-blue-600">
              {(summary.siglip_accuracy * 100).toFixed(2)}%
            </p>
          </div>
          <div className="bg-green-50 p-6 rounded-lg border-2 border-green-200 text-center">
            <p className="text-sm text-gray-600 mb-2">ViT Accuracy</p>
            <p className="text-4xl font-bold text-green-600">
              {(summary.vit_accuracy * 100).toFixed(2)}%
            </p>
          </div>
          <div className="bg-purple-50 p-6 rounded-lg border-2 border-purple-200 text-center">
            <p className="text-sm text-gray-600 mb-2">Ensemble Accuracy</p>
            <p className="text-4xl font-bold text-purple-600">
              {(summary.ensemble_accuracy * 100).toFixed(2)}%
            </p>
            <p className="text-sm text-gray-600 mt-1">
              AUC: {summary.ensemble_auc.toFixed(3)}
            </p>
          </div>
        </div>

        {/* Performance Comparison Table (Table 9) */}
        <div className="mb-8">
          <h3 className="text-xl font-bold mb-4">
            📈 Performance Comparison Table (Table 9)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse border border-gray-300">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border border-gray-300 px-4 py-2 text-left">
                    Model
                  </th>
                  <th className="border border-gray-300 px-4 py-2">Accuracy</th>
                  <th className="border border-gray-300 px-4 py-2">
                    Precision
                  </th>
                  <th className="border border-gray-300 px-4 py-2">Recall</th>
                  <th className="border border-gray-300 px-4 py-2">F1-Score</th>
                  <th className="border border-gray-300 px-4 py-2">AUC</th>
                </tr>
              </thead>
              <tbody>
                {performance_comparison.models.map((model, idx) => (
                  <tr
                    key={idx}
                    className={
                      idx === performance_comparison.models.length - 1
                        ? "bg-purple-50 font-semibold"
                        : ""
                    }
                  >
                    <td className="border border-gray-300 px-4 py-2">
                      {model}
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center">
                      {(performance_comparison.accuracy[idx] * 100).toFixed(2)}%
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center">
                      {(performance_comparison.precision[idx] * 100).toFixed(2)}
                      %
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center">
                      {(performance_comparison.recall[idx] * 100).toFixed(2)}%
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center">
                      {(performance_comparison.f1_score[idx] * 100).toFixed(2)}%
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center">
                      {performance_comparison.auc[idx].toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Performance Comparison Chart */}
        <div className="mb-8">
          <h3 className="text-xl font-bold mb-4">
            Performance Metrics Visualization
          </h3>
          <img
            src="http://localhost:8000/api/metrics/image/performance_comparison.png"
            alt="Performance Comparison"
            className="w-full rounded-lg shadow-lg border"
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
        </div>

        {/* Class-wise Performance (Table 2) */}
        <div className="mb-8">
          <h3 className="text-xl font-bold mb-4">
            📋 Class-wise Performance (Table 2)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse border border-gray-300">
              <thead className="bg-gray-100">
                <tr>
                  <th
                    className="border border-gray-300 px-4 py-2 text-left"
                    rowSpan="2"
                  >
                    Class
                  </th>
                  <th className="border border-gray-300 px-4 py-2" colSpan="3">
                    SigLIP
                  </th>
                  <th className="border border-gray-300 px-4 py-2" colSpan="3">
                    ViT
                  </th>
                  <th className="border border-gray-300 px-4 py-2" colSpan="3">
                    Ensemble
                  </th>
                </tr>
                <tr>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    Precision
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    Recall
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    F1
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    Precision
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    Recall
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    F1
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    Precision
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    Recall
                  </th>
                  <th className="border border-gray-300 px-2 py-1 text-xs">
                    F1
                  </th>
                </tr>
              </thead>
              <tbody>
                {classwise_performance.class.map((className, idx) => (
                  <tr key={idx}>
                    <td className="border border-gray-300 px-4 py-2 font-semibold">
                      {className}
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm">
                      {(
                        classwise_performance.siglip_precision[idx] * 100
                      ).toFixed(1)}
                      %
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm">
                      {(classwise_performance.siglip_recall[idx] * 100).toFixed(
                        1
                      )}
                      %
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm">
                      {(classwise_performance.siglip_f1[idx] * 100).toFixed(1)}%
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm">
                      {(classwise_performance.vit_precision[idx] * 100).toFixed(
                        1
                      )}
                      %
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm">
                      {(classwise_performance.vit_recall[idx] * 100).toFixed(1)}
                      %
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm">
                      {(classwise_performance.vit_f1[idx] * 100).toFixed(1)}%
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm bg-purple-50">
                      {(
                        classwise_performance.ensemble_precision[idx] * 100
                      ).toFixed(1)}
                      %
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm bg-purple-50">
                      {(
                        classwise_performance.ensemble_recall[idx] * 100
                      ).toFixed(1)}
                      %
                    </td>
                    <td className="border border-gray-300 px-2 py-2 text-center text-sm bg-purple-50">
                      {(classwise_performance.ensemble_f1[idx] * 100).toFixed(
                        1
                      )}
                      %
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Confusion Matrices */}
        <div className="mb-8">
          <h3 className="text-xl font-bold mb-4">
            🔲 Confusion Matrices (Figure 5 & 10)
          </h3>
          <img
            src="http://localhost:8000/api/metrics/image/confusion_matrices_all.png"
            alt="Confusion Matrices"
            className="w-full rounded-lg shadow-lg border"
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
        </div>

        {/* ROC Curves */}
        <div className="mb-8">
          <h3 className="text-xl font-bold mb-4">📉 ROC Curves (Figure 11)</h3>
          <img
            src="http://localhost:8000/api/metrics/image/roc_curves.png"
            alt="ROC Curves"
            className="w-full rounded-lg shadow-lg border"
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
        </div>

        {/* Dataset Distribution (Table 8) */}
        <div className="mb-8">
          <h3 className="text-xl font-bold mb-4">
            📊 Dataset Distribution (Table 8)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse border border-gray-300">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border border-gray-300 px-4 py-2 text-left">
                    Split
                  </th>
                  <th className="border border-gray-300 px-4 py-2">
                    No Fracture
                  </th>
                  <th className="border border-gray-300 px-4 py-2">Fracture</th>
                  <th className="border border-gray-300 px-4 py-2">Total</th>
                </tr>
              </thead>
              <tbody>
                {dataset_distribution.split.map((split, idx) => (
                  <tr key={idx}>
                    <td className="border border-gray-300 px-4 py-2 font-semibold">
                      {split}
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center">
                      {dataset_distribution.no_fracture[idx]}
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center">
                      {dataset_distribution.fracture[idx]}
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-center font-semibold">
                      {dataset_distribution.total[idx]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Fuzzy Logic Information */}
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg border border-purple-300">
          <h3 className="text-xl font-bold mb-4">
            🔮 Fuzzy Logic Risk Assessment (Table 3)
          </h3>
          <p className="text-gray-700 mb-4">
            Our system implements 5-level fuzzy logic to translate numeric
            predictions into clinical risk categories:
          </p>
          <div className="grid grid-cols-5 gap-3">
            <div className="bg-blue-100 p-4 rounded-lg text-center">
              <p className="font-semibold text-blue-800">Very Low Risk</p>
              <p className="text-sm text-gray-600">0-20%</p>
            </div>
            <div className="bg-green-100 p-4 rounded-lg text-center">
              <p className="font-semibold text-green-800">Low Risk</p>
              <p className="text-sm text-gray-600">20-40%</p>
            </div>
            <div className="bg-yellow-100 p-4 rounded-lg text-center">
              <p className="font-semibold text-yellow-800">Medium Risk</p>
              <p className="text-sm text-gray-600">40-60%</p>
            </div>
            <div className="bg-orange-100 p-4 rounded-lg text-center">
              <p className="font-semibold text-orange-800">High Risk</p>
              <p className="text-sm text-gray-600">60-80%</p>
            </div>
            <div className="bg-red-100 p-4 rounded-lg text-center">
              <p className="font-semibold text-red-800">Very High Risk</p>
              <p className="text-sm text-gray-600">80-100%</p>
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4">
            This fuzzy inference system enhances clinical interpretability by
            providing linguistic risk levels that align with real-world
            diagnostic language.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ModelPerformance;
