import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const ModelPerformance = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        "http://localhost:8000/api/metrics/all-data"
      );
      setData(response.data);
      setError(null);
    } catch (err) {
      setError("Failed to load metrics. Please run evaluate_model.py first.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-50">
        <div className="text-center">
          <div className="text-7xl mb-6 animate-pulse">📊</div>
          <p className="text-3xl font-bold text-gray-800 mb-3">
            Loading Performance Data...
          </p>
          <p className="text-gray-600 text-lg">This may take a moment</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8 bg-gradient-to-br from-red-50 to-orange-50">
        <div className="max-w-2xl mx-auto">
          <div className="bg-white border-4 border-red-400 rounded-3xl p-12 text-center shadow-2xl">
            <div className="text-7xl mb-6">❌</div>
            <h2 className="text-4xl font-bold text-red-800 mb-5">{error}</h2>
            <div className="space-y-4">
              <button
                onClick={fetchMetrics}
                className="bg-red-600 text-white px-10 py-4 rounded-xl font-bold hover:bg-red-700 transition text-lg"
              >
                Retry Loading
              </button>
              <br />
              <button
                onClick={() => navigate("/")}
                className="text-gray-700 hover:text-gray-900 transition font-semibold text-lg"
              >
                ← Back to Home
              </button>
            </div>
            <div className="mt-8 bg-gray-100 p-6 rounded-2xl text-left">
              <p className="font-bold text-gray-800 mb-3 text-lg">
                To generate metrics:
              </p>
              <code className="bg-gray-800 text-green-400 px-4 py-2 rounded block">
                cd backend && python3 evaluate_model.py
              </code>
            </div>
          </div>
        </div>
      </div>
    );
  }

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

  return (
    <div className="min-h-screen p-8 bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate("/")}
            className="mb-6 flex items-center text-gray-700 hover:text-gray-900 transition font-bold text-lg bg-white px-6 py-3 rounded-xl shadow-md hover:shadow-lg"
          >
            ← Back to Home
          </button>

          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl shadow-2xl p-10 text-white">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-5xl font-bold mb-4">
                  📊 Model Performance Analysis
                </h1>
                <p className="text-xl text-blue-100">
                  Comprehensive evaluation of transformer models on fracture
                  detection
                </p>
              </div>
              <div className="text-8xl">🏆</div>
            </div>

            {/* Summary Stats - FIXED COLORS */}
            {data?.summary && (
              <div className="mt-8 grid md:grid-cols-4 gap-6">
                <div className="bg-white rounded-2xl p-6 shadow-lg">
                  <div className="text-5xl mb-3">📸</div>
                  <div className="text-4xl font-bold text-gray-800">
                    {data.summary.total_images}
                  </div>
                  <div className="text-sm text-gray-600 font-semibold mt-1">
                    Total Images
                  </div>
                </div>
                <div className="bg-white rounded-2xl p-6 shadow-lg">
                  <div className="text-5xl mb-3">🎯</div>
                  <div className="text-4xl font-bold text-gray-800">
                    {(data.summary.ensemble_accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600 font-semibold mt-1">
                    Ensemble Accuracy
                  </div>
                </div>
                <div className="bg-white rounded-2xl p-6 shadow-lg">
                  <div className="text-5xl mb-3">📈</div>
                  <div className="text-4xl font-bold text-gray-800">
                    {data.summary.ensemble_auc.toFixed(3)}
                  </div>
                  <div className="text-sm text-gray-600 font-semibold mt-1">
                    AUC Score
                  </div>
                </div>
                <div className="bg-white rounded-2xl p-6 shadow-lg">
                  <div className="text-5xl mb-3">🤖</div>
                  <div className="text-4xl font-bold text-gray-800">2</div>
                  <div className="text-sm text-gray-600 font-semibold mt-1">
                    Models Tested
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-10">
          {/* FIGURE 2 */}
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
              <strong>Description:</strong> Training and validation
              loss/accuracy for SigLIP and ViT models. Both converge within 7
              epochs, demonstrating efficient optimization.
            </p>
          </section>

          {/* FIGURE 3 */}
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
            <p className="mt-4 text-gray-600 text-sm italic bg-blue-50 p-4 rounded-lg">
              <strong>Description:</strong> Ensemble consistently outperforms
              individual models across all metrics.
            </p>
          </section>

          {/* TABLE 1 */}
          {data?.table1_performance_comparison && (
            <section className="bg-gradient-to-br from-white to-blue-50 rounded-2xl shadow-xl p-8 border-2 border-blue-200">
              <TableComponent
                title="Table 1: Performance Comparisons for All Models"
                data={data.table1_performance_comparison}
                description="Comprehensive metrics including Kappa and MCC. Ensemble achieves highest scores across all metrics."
              />
            </section>
          )}

          {/* FIGURE 4 */}
          <section className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-200">
            <h2 className="text-3xl font-bold mb-6 flex items-center">
              <span className="text-4xl mr-3">🎯</span>
              Figure 4: Per-Class F1-Score Comparison
            </h2>
            <img
              src={`http://localhost:8000/api/metrics/image/figure_4_perclass_f1.png`}
              alt="Per-Class F1"
              className="w-full rounded-xl shadow-lg"
              onError={(e) => {
                e.target.style.display = "none";
              }}
            />
            <p className="mt-4 text-gray-600 text-sm italic bg-blue-50 p-4 rounded-lg">
              <strong>Description:</strong> Balanced performance on both
              classes. Ensemble reduces class imbalance.
            </p>
          </section>

          {/* TABLE 2 - FIXED */}
          {data?.classwise_performance && (
            <section className="bg-gradient-to-br from-white to-green-50 rounded-2xl shadow-xl p-8 border-2 border-green-200">
              <TableComponent
                title="Table 2: Class-wise Performance"
                data={data.classwise_performance}
                description="Detailed precision, recall, and F1-score for each class. Both classes achieve >90% metrics."
              />
            </section>
          )}

          {/* FIGURE 5 */}
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
            <p className="mt-4 text-gray-600 text-sm italic bg-blue-50 p-4 rounded-lg">
              <strong>Description:</strong> Ensemble reduces misclassification
              compared to individual models.
            </p>
          </section>

          {/* TABLE 3-7 */}
          {data?.table3_fuzzy_granularity && (
            <section className="bg-gradient-to-br from-white to-purple-50 rounded-2xl shadow-xl p-8 border-2 border-purple-200">
              <TableComponent
                title="Table 3: Fuzzy Granularity and Interpretability"
                data={data.table3_fuzzy_granularity}
                description="5-level system provides higher accuracy and clinical interpretability."
              />
            </section>
          )}

          {data?.table4_fusion_strategies && (
            <section className="bg-gradient-to-br from-white to-orange-50 rounded-2xl shadow-xl p-8 border-2 border-orange-200">
              <TableComponent
                title="Table 4: Ensemble Fusion Strategies"
                data={data.table4_fusion_strategies}
                description="Meta-learner outperforms simple averaging by learning optimal weights."
              />
            </section>
          )}

          {/* FIGURE 7 */}
          <section className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-200">
            <h2 className="text-3xl font-bold mb-6 flex items-center">
              <span className="text-4xl mr-3">🕸️</span>
              Figure 7: Radar Chart
            </h2>
            <div className="flex justify-center">
              <img
                src={`http://localhost:8000/api/metrics/image/figure_7_radar_chart.png`}
                alt="Radar Chart"
                className="max-w-2xl w-full rounded-xl shadow-lg"
                onError={(e) => {
                  e.target.style.display = "none";
                }}
              />
            </div>
            <p className="mt-4 text-gray-600 text-sm italic bg-blue-50 p-4 rounded-lg">
              <strong>Description:</strong> Ensemble shows largest coverage,
              indicating superior overall performance.
            </p>
          </section>

          {data?.table5_early_stopping && (
            <section className="bg-gradient-to-br from-white to-cyan-50 rounded-2xl shadow-xl p-8 border-2 border-cyan-200">
              <TableComponent
                title="Table 5: Early Stopping and Convergence"
                data={data.table5_early_stopping}
                description="Models converge within 5-7 epochs, preventing overfitting."
              />
            </section>
          )}

          {data?.table6_gradcam_info && (
            <section className="bg-gradient-to-br from-white to-pink-50 rounded-2xl shadow-xl p-8 border-2 border-pink-200">
              <TableComponent
                title="Table 6: Grad-CAM Visualizations"
                data={data.table6_gradcam_info}
                description="ViT uses attention rollout, SigLIP employs Grad-CAM for explainability."
              />
            </section>
          )}

          {data?.table7_attention_strength && (
            <section className="bg-gradient-to-br from-white to-yellow-50 rounded-2xl shadow-xl p-8 border-2 border-yellow-200">
              <TableComponent
                title="Table 7: Attention Strength Across Layers"
                data={data.table7_attention_strength}
                description="Mid-layers (3-5) capture most diagnostically relevant features."
              />
            </section>
          )}

          {/* TABLE 8 - FIXED */}
          {data?.dataset_distribution && (
            <section className="bg-gradient-to-br from-white to-indigo-50 rounded-2xl shadow-xl p-8 border-2 border-indigo-200">
              <TableComponent
                title="Table 8: Dataset Distribution"
                data={data.dataset_distribution}
                description="Train/validation/test split with balanced class distribution across all splits."
              />
            </section>
          )}

          {/* FIGURE 11 & 12 */}
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
            <p className="mt-4 text-gray-600 text-sm italic bg-blue-50 p-4 rounded-lg">
              <strong>Description:</strong> Ensemble achieves AUC ≈ 0.98,
              excellent discrimination.
            </p>
          </section>

          <section className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-200">
            <h2 className="text-3xl font-bold mb-6 flex items-center">
              <span className="text-4xl mr-3">🎚️</span>
              Figure 12: Precision-Recall Curves
            </h2>
            <img
              src={`http://localhost:8000/api/metrics/image/figure_12_precision_recall.png`}
              alt="Precision-Recall"
              className="w-full rounded-xl shadow-lg"
              onError={(e) => {
                e.target.style.display = "none";
              }}
            />
            <p className="mt-4 text-gray-600 text-sm italic bg-blue-50 p-4 rounded-lg">
              <strong>Description:</strong> Ensemble maintains high precision
              across all recall levels.
            </p>
          </section>

          {/* FINAL SUMMARY - FIXED COLORS */}
          {data?.performance_comparison && (
            <section className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-3xl shadow-2xl p-10 text-white">
              <h2 className="text-4xl font-bold mb-8 flex items-center">
                <span className="text-5xl mr-4">🏆</span>
                Final Performance Summary
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                {data.performance_comparison.models.map((model, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-2xl p-6 shadow-xl text-gray-800"
                  >
                    <h3 className="text-2xl font-bold mb-4 text-emerald-700">
                      {model}
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="font-semibold">Accuracy:</span>
                        <span className="font-bold text-emerald-600">
                          {(
                            data.performance_comparison.accuracy[idx] * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-semibold">Precision:</span>
                        <span className="font-bold text-emerald-600">
                          {(
                            data.performance_comparison.precision[idx] * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-semibold">Recall:</span>
                        <span className="font-bold text-emerald-600">
                          {(
                            data.performance_comparison.recall[idx] * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-semibold">F1-Score:</span>
                        <span className="font-bold text-emerald-600">
                          {(
                            data.performance_comparison.f1_score[idx] * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="flex justify-between border-t-2 border-gray-300 pt-2 mt-2">
                        <span className="font-bold">AUC:</span>
                        <span className="font-bold text-2xl text-emerald-700">
                          {data.performance_comparison.auc[idx].toFixed(3)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {data?.summary && (
            <section className="bg-gray-200 rounded-2xl p-6 text-sm text-gray-700">
              <p className="text-center font-semibold">
                <strong>Evaluation completed:</strong>{" "}
                {data.summary.generated_at} •<strong> Dataset:</strong>{" "}
                {data.summary.total_images} images •<strong> Models:</strong>{" "}
                {data.summary.models_used.join(", ")}
              </p>
            </section>
          )}
        </div>

        <div className="text-center mt-12 pb-8">
          <button
            onClick={() => navigate("/")}
            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-12 py-5 rounded-2xl font-bold text-xl hover:from-blue-700 hover:to-purple-700 transition shadow-xl"
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelPerformance;
