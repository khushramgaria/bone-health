import React from "react";

const ModelInfo = () => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6 border border-gray-200">
      <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b pb-3">
        🔧 Model Configuration & Hyperparameters
      </h2>

      {/* Models Used */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-700 flex items-center">
          <span className="mr-2">🤖</span> Models Used
        </h3>
        <div className="flex gap-2 flex-wrap">
          <span className="px-4 py-2 bg-blue-100 text-blue-800 rounded-lg font-medium">
            SigLIP
          </span>
          <span className="px-4 py-2 bg-green-100 text-green-800 rounded-lg font-medium">
            ViT (Vision Transformer)
          </span>
          <span className="px-4 py-2 bg-purple-100 text-purple-800 rounded-lg font-medium">
            Ensemble
          </span>
        </div>
      </div>

      {/* Optimizer Configuration */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-700 flex items-center">
          <span className="mr-2">⚙️</span> Optimizer Configuration
        </h3>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Type:</span>
              <span className="font-semibold text-gray-800">AdamW</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Learning Rate:</span>
              <span className="font-semibold text-gray-800">2e-5 to 5e-5</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Weight Decay:</span>
              <span className="font-semibold text-gray-800">0.01</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Scheduler:</span>
              <span className="font-semibold text-gray-800">
                Linear with Warmup
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Beta 1:</span>
              <span className="font-semibold text-gray-800">0.9</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Beta 2:</span>
              <span className="font-semibold text-gray-800">0.999</span>
            </div>
          </div>
        </div>
      </div>

      {/* Hyperparameters */}
      <div>
        <h3 className="text-lg font-semibold mb-3 text-gray-700 flex items-center">
          <span className="mr-2">📊</span> Training Hyperparameters
        </h3>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">
                Batch Size (Train):
              </span>
              <span className="font-semibold text-gray-800">16</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">
                Batch Size (Val):
              </span>
              <span className="font-semibold text-gray-800">32</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Epochs:</span>
              <span className="font-semibold text-gray-800">7</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Image Size:</span>
              <span className="font-semibold text-gray-800">224×224</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Dropout Rate:</span>
              <span className="font-semibold text-gray-800">0.1</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Warmup Steps:</span>
              <span className="font-semibold text-gray-800">50-100</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Loss Function:</span>
              <span className="font-semibold text-gray-800">CrossEntropy</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">
                Mixed Precision:
              </span>
              <span className="font-semibold text-gray-800">FP16</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">
                Gradient Clipping:
              </span>
              <span className="font-semibold text-gray-800">
                Max Norm = 1.0
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Early Stopping:</span>
              <span className="font-semibold text-gray-800">Patience = 3</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Random Seed:</span>
              <span className="font-semibold text-gray-800">42</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">
                Fuzzy Logic Levels:
              </span>
              <span className="font-semibold text-gray-800">5</span>
            </div>
          </div>
        </div>
      </div>

      {/* Training Environment */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-700 flex items-center">
          <span className="mr-2">💻</span> Training Environment
        </h3>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Framework:</span>
              <span className="font-semibold text-gray-800">PyTorch 2.0+</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Transformers:</span>
              <span className="font-semibold text-gray-800">SigLIP, ViT</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Hardware:</span>
              <span className="font-semibold text-gray-800">
                NVIDIA GPU / CPU
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">GPU Memory:</span>
              <span className="font-semibold text-gray-800">~8GB</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;
