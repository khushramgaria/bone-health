import React, { useState, useEffect } from "react";
import axios from "axios";

const BoneHealthReport = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const response = await axios.get(
        "http://localhost:8000/api/bone-health/report"
      );
      console.log("Bone health data:", response.data); // Debug log
      setData(response.data);
      setLoading(false);
    } catch (err) {
      console.error("Error loading bone health data:", err);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-600 to-teal-600 flex items-center justify-center">
        <div className="text-white text-center">
          <div className="animate-spin rounded-full h-20 w-20 border-b-4 border-white mx-auto mb-4"></div>
          <p className="text-2xl font-semibold">
            Loading Bone Health Report...
          </p>
        </div>
      </div>
    );
  }

  // FIX: Properly extract summary data
  const summary = data?.summary || {};
  const report = data?.report || {};

  // Calculate metrics with fallbacks
  const accuracy = summary.overall_accuracy || 0;
  const precision = report?.["weighted avg"]?.precision
    ? report["weighted avg"].precision * 100
    : 0;
  const recall = report?.["weighted avg"]?.recall
    ? report["weighted avg"].recall * 100
    : 0;
  const f1Score = report?.["weighted avg"]?.["f1-score"]
    ? report["weighted avg"]["f1-score"] * 100
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-600 via-teal-600 to-blue-500 py-8 px-4">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="bg-gradient-to-r from-green-900 to-teal-700 rounded-2xl shadow-2xl p-8 text-white">
          <h1 className="text-5xl font-bold mb-2">
            🦴 Bone Health Detection System
          </h1>
          <p className="text-xl opacity-90">
            DEXA Scan Analysis - Osteoporosis Detection
          </p>
          <p className="text-sm mt-4 opacity-75">
            Generated: {summary.generated_at || "N/A"}
          </p>
          <p className="text-sm opacity-75">
            Model: {summary.model || "BiomedCLIP"}
          </p>
        </div>

        {/* Executive Summary */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-3xl font-bold text-green-900 mb-4">
            📊 Executive Summary
          </h2>
          <p className="text-gray-700 text-lg leading-relaxed mb-6">
            Comprehensive bone health evaluation using{" "}
            {summary.model || "BiomedCLIP"} deep learning model. Analysis of{" "}
            {summary.total_images || 0} DEXA spine scans across three
            categories: Normal bone density, Osteopenia (low bone mass), and
            Osteoporosis (severe bone loss).
          </p>

          {/* Metric Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <MetricCard
              title="Overall Accuracy"
              value={`${accuracy.toFixed(1)}%`}
              color="from-green-500 to-green-700"
            />
            <MetricCard
              title="Precision"
              value={`${precision.toFixed(1)}%`}
              color="from-teal-500 to-teal-700"
            />
            <MetricCard
              title="Recall"
              value={`${recall.toFixed(1)}%`}
              color="from-blue-500 to-blue-700"
            />
            <MetricCard
              title="F1-Score"
              value={`${f1Score.toFixed(1)}%`}
              color="from-purple-500 to-purple-700"
            />
          </div>
        </div>

        {/* Dataset Overview */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-3xl font-bold text-green-900 mb-6">
            📁 Dataset Overview
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-green-50 p-6 rounded-lg border-l-4 border-green-500">
              <h3 className="font-bold text-xl mb-2 text-green-900">Normal</h3>
              <p className="text-4xl font-bold text-green-600">
                {summary.per_class_counts?.normal || "0/0"}
              </p>
              <p className="text-gray-600 mt-2">
                Healthy bone density (T-score &gt; -1.0)
              </p>
              <p className="text-sm text-green-700 font-semibold mt-2">
                Accuracy: {summary.per_class_accuracy?.normal?.toFixed(1) || 0}%
              </p>
            </div>

            <div className="bg-yellow-50 p-6 rounded-lg border-l-4 border-yellow-500">
              <h3 className="font-bold text-xl mb-2 text-yellow-900">
                Osteopenia
              </h3>
              <p className="text-4xl font-bold text-yellow-600">
                {summary.per_class_counts?.osteopenia || "0/0"}
              </p>
              <p className="text-gray-600 mt-2">
                Low bone mass (T-score -1.0 to -2.5)
              </p>
              <p className="text-sm text-yellow-700 font-semibold mt-2">
                Accuracy:{" "}
                {summary.per_class_accuracy?.osteopenia?.toFixed(1) || 0}%
              </p>
            </div>

            <div className="bg-red-50 p-6 rounded-lg border-l-4 border-red-500">
              <h3 className="font-bold text-xl mb-2 text-red-900">
                Osteoporosis
              </h3>
              <p className="text-4xl font-bold text-red-600">
                {summary.per_class_counts?.osteoporosis || "0/0"}
              </p>
              <p className="text-gray-600 mt-2">
                Severe bone loss (T-score &lt; -2.5)
              </p>
              <p className="text-sm text-red-700 font-semibold mt-2">
                Accuracy:{" "}
                {summary.per_class_accuracy?.osteoporosis?.toFixed(1) || 0}%
              </p>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-3xl font-bold text-green-900 mb-6">
            🎯 Per-Class Performance
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            {["Normal", "Osteopenia", "Osteoporosis"].map((category, idx) => (
              <PerformanceCard
                key={idx}
                title={category}
                data={report?.[category]}
                color={
                  idx === 0
                    ? "from-green-400 to-green-600"
                    : idx === 1
                    ? "from-yellow-400 to-yellow-600"
                    : "from-red-400 to-red-600"
                }
              />
            ))}
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-3xl font-bold text-green-900 mb-4">
            📊 Confusion Matrix
          </h2>
          <p className="text-gray-600 mb-6 italic">
            Visual representation of classification accuracy
          </p>

          <div className="flex justify-center">
            <img
              src="http://localhost:8000/api/bone-health/images/figure_1_confusion_matrix.png"
              alt="Confusion Matrix"
              className="max-w-full h-auto rounded-lg shadow-lg"
            />
          </div>
        </div>

        {/* Per-Class Chart */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-3xl font-bold text-green-900 mb-4">
            📈 Per-Class Metrics Comparison
          </h2>

          <div className="flex justify-center">
            <img
              src="http://localhost:8000/api/bone-health/images/figure_2_perclass_performance.png"
              alt="Per-Class Performance"
              className="max-w-full h-auto rounded-lg shadow-lg"
            />
          </div>
        </div>

        {/* Clinical Guidelines */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-3xl font-bold text-green-900 mb-6">
            🏥 Clinical Guidelines (WHO Standards)
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="bg-green-900 text-white">
                <tr>
                  <th className="py-3 px-4 text-left">Category</th>
                  <th className="py-3 px-4 text-left">T-Score Range</th>
                  <th className="py-3 px-4 text-left">Risk Level</th>
                  <th className="py-3 px-4 text-left">Recommendations</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                <tr className="hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full font-bold">
                      🟢 Normal
                    </span>
                  </td>
                  <td className="py-3 px-4">Above -1.0</td>
                  <td className="py-3 px-4">Low</td>
                  <td className="py-3 px-4">
                    Maintain healthy lifestyle, regular exercise
                  </td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full font-bold">
                      🟡 Osteopenia
                    </span>
                  </td>
                  <td className="py-3 px-4">-1.0 to -2.5</td>
                  <td className="py-3 px-4">Moderate</td>
                  <td className="py-3 px-4">
                    Calcium/Vitamin D supplementation, weight-bearing exercise
                  </td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full font-bold">
                      🔴 Osteoporosis
                    </span>
                  </td>
                  <td className="py-3 px-4">Below -2.5</td>
                  <td className="py-3 px-4">High</td>
                  <td className="py-3 px-4">
                    Medical intervention, medication, fall prevention
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-800 rounded-2xl shadow-2xl p-8 text-white text-center">
          <p className="text-xl font-semibold">
            © 2025 Bone Health Detection System
          </p>
          <p className="mt-2 opacity-75">
            DEXA Scan Analysis - Powered by {summary.model || "AI"}
          </p>
          <p className="mt-2 text-sm opacity-60">
            Total Images Analyzed: {summary.total_images || 0} | Correct
            Predictions: {summary.correct_predictions || 0} | Overall Accuracy:{" "}
            {accuracy.toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
};

// Helper Components
const MetricCard = ({ title, value, color }) => (
  <div
    className={`bg-gradient-to-br ${color} rounded-xl shadow-lg p-6 text-white`}
  >
    <p className="text-sm opacity-90 mb-2">{title}</p>
    <p className="text-4xl font-bold">{value}</p>
  </div>
);

const PerformanceCard = ({ title, data, color }) => {
  if (!data) {
    return (
      <div
        className={`bg-gradient-to-br ${color} rounded-xl shadow-lg p-6 text-white`}
      >
        <h3 className="text-2xl font-bold mb-3 text-center">{title}</h3>
        <p className="text-center">No data available</p>
      </div>
    );
  }

  return (
    <div
      className={`bg-gradient-to-br ${color} rounded-xl shadow-lg p-6 text-white`}
    >
      <h3 className="text-2xl font-bold mb-3 text-center">{title}</h3>
      <div className="space-y-2">
        <div className="flex justify-between">
          <span>Precision:</span>
          <span className="font-bold">
            {((data.precision || 0) * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span>Recall:</span>
          <span className="font-bold">
            {((data.recall || 0) * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span>F1-Score:</span>
          <span className="font-bold">
            {((data["f1-score"] || 0) * 100).toFixed(1)}%
          </span>
        </div>
        <div className="bg-white bg-opacity-20 rounded p-2 mt-3">
          <div className="flex justify-between">
            <span>Support:</span>
            <span className="font-bold">{data.support || 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BoneHealthReport;
