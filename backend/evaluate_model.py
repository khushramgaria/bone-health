import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from io import BytesIO
import time

load_dotenv()

print("Starting Model Evaluation (SigLIP + Real Gemini as ViT)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SigLIP
siglip_processor = AutoImageProcessor.from_pretrained("prithivMLmods/Bone-Fracture-Detection")
siglip_model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Bone-Fracture-Detection").to(device)
siglip_model.eval()
print("✓ SigLIP loaded")

# Configure Gemini (as ViT)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("✓ Gemini (ViT) configured")

# Test dataset paths
test_data_path = "test_dataset/"
fracture_folder = os.path.join(test_data_path, "fracture/")
no_fracture_folder = os.path.join(test_data_path, "no_fracture/")

if not os.path.exists(fracture_folder) or not os.path.exists(no_fracture_folder):
    print(f"❌ Error: Please create test dataset folders")
    exit()

# Arrays for predictions
y_true = []
siglip_probs = []
vit_probs = []
ensemble_probs = []


def get_gemini_prediction(image):
    """Get real Gemini prediction for fracture detection"""
    try:
        # Convert PIL image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        prompt = """Analyze this X-ray for bone fractures.

Respond ONLY with a number between 0 and 100 representing your confidence that a fracture is present.
- 0 = Definitely no fracture
- 50 = Uncertain
- 100 = Definitely fracture present

Just respond with the number, nothing else."""

        response = gemini_model.generate_content([
            prompt, 
            {"mime_type": "image/png", "data": img_byte_arr}
        ])
        
        # Parse response
        response_text = response.text.strip()
        
        # Extract number
        import re
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            confidence = int(numbers[0])
            confidence = np.clip(confidence, 0, 100)
            return confidence / 100.0  # Convert to probability
        else:
            return 0.5  # Default uncertain
            
    except Exception as e:
        print(f"  Gemini error: {e}")
        return 0.5  # Default if error


print("\n⚠️  Using REAL Gemini predictions - This will be slower (5-10 seconds per image)")
print("Processing fracture images...")

fracture_images = [f for f in os.listdir(fracture_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))]
for i, img_name in enumerate(fracture_images):
    try:
        img_path = os.path.join(fracture_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # SigLIP prediction
        print(f"  [{i+1}/{len(fracture_images)}] SigLIP analyzing {img_name}...")
        inputs = siglip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = siglip_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            siglip_prob = probs[0][1].item()
        
        print(f"      SigLIP: {siglip_prob*100:.1f}%")
        
        # Real Gemini (ViT) prediction
        print(f"      Gemini analyzing...")
        vit_prob = get_gemini_prediction(img)
        print(f"      Gemini: {vit_prob*100:.1f}%")
        
        # Ensemble
        ensemble_prob = (siglip_prob + vit_prob) / 2
        
        y_true.append(1)
        siglip_probs.append(siglip_prob)
        vit_probs.append(vit_prob)
        ensemble_probs.append(ensemble_prob)
        
        time.sleep(1)  # Small delay to avoid rate limits
        
    except Exception as e:
        print(f"  Error processing {img_name}: {e}")

print("\nProcessing no fracture images...")
no_fracture_images = [f for f in os.listdir(no_fracture_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))]
for i, img_name in enumerate(no_fracture_images):
    try:
        img_path = os.path.join(no_fracture_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        
        print(f"  [{i+1}/{len(no_fracture_images)}] SigLIP analyzing {img_name}...")
        inputs = siglip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = siglip_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            siglip_prob = probs[0][1].item()
        
        print(f"      SigLIP: {siglip_prob*100:.1f}%")
        
        print(f"      Gemini analyzing...")
        vit_prob = get_gemini_prediction(img)
        print(f"      Gemini: {vit_prob*100:.1f}%")
        
        ensemble_prob = (siglip_prob + vit_prob) / 2
        
        y_true.append(0)
        siglip_probs.append(siglip_prob)
        vit_probs.append(vit_prob)
        ensemble_probs.append(ensemble_prob)
        
        time.sleep(1)
        
    except Exception as e:
        print(f"  Error processing {img_name}: {e}")

print(f"\n✓ Total images processed: {len(y_true)}")

# Show prediction comparison
print("\n" + "="*60)
print("PREDICTION COMPARISON:")
print("="*60)
for i in range(len(y_true)):
    actual = "Fracture" if y_true[i] == 1 else "No Fracture"
    print(f"Image {i+1} (Actual: {actual})")
    print(f"  SigLIP:   {siglip_probs[i]*100:.1f}%")
    print(f"  Gemini:   {vit_probs[i]*100:.1f}%")
    print(f"  Ensemble: {ensemble_probs[i]*100:.1f}%")
    print()

# Convert probabilities to predictions
siglip_pred = [1 if p > 0.5 else 0 for p in siglip_probs]
vit_pred = [1 if p > 0.5 else 0 for p in vit_probs]
ensemble_pred = [1 if p > 0.5 else 0 for p in ensemble_probs]

# Create output folder
output_folder = "metrics_output/"
os.makedirs(output_folder, exist_ok=True)

# ============ CONFUSION MATRICES ============
print("\nGenerating Confusion Matrices...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# SigLIP Confusion Matrix
cm_siglip = confusion_matrix(y_true, siglip_pred)
sns.heatmap(cm_siglip, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Fracture', 'Fracture'],
            yticklabels=['No Fracture', 'Fracture'])
axes[0].set_title('SigLIP Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# ViT (Gemini) Confusion Matrix
cm_vit = confusion_matrix(y_true, vit_pred)
sns.heatmap(cm_vit, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Fracture', 'Fracture'],
            yticklabels=['No Fracture', 'Fracture'])
axes[1].set_title('ViT (Gemini) Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# Ensemble Confusion Matrix
cm_ensemble = confusion_matrix(y_true, ensemble_pred)
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Purples', ax=axes[2],
            xticklabels=['No Fracture', 'Fracture'],
            yticklabels=['No Fracture', 'Fracture'])
axes[2].set_title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'confusion_matrices_all.png'), dpi=150)
plt.close()
print("✓ Confusion matrices saved")

# ============ CLASSIFICATION REPORTS ============
print("Generating Classification Reports...")

siglip_report = classification_report(y_true, siglip_pred, target_names=['No Fracture', 'Fracture'], output_dict=True)
vit_report = classification_report(y_true, vit_pred, target_names=['No Fracture', 'Fracture'], output_dict=True)
ensemble_report = classification_report(y_true, ensemble_pred, target_names=['No Fracture', 'Fracture'], output_dict=True)

with open(os.path.join(output_folder, 'siglip_report.json'), 'w') as f:
    json.dump(siglip_report, f, indent=4)

with open(os.path.join(output_folder, 'vit_report.json'), 'w') as f:
    json.dump(vit_report, f, indent=4)

with open(os.path.join(output_folder, 'ensemble_report.json'), 'w') as f:
    json.dump(ensemble_report, f, indent=4)

print("✓ Classification reports saved")

# ============ PERFORMANCE COMPARISON TABLE ============
print("Generating Performance Comparison Table...")

models_data = {
    'SigLIP': siglip_report,
    'ViT': vit_report,
    'Ensemble': ensemble_report
}

performance_table = {
    'models': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'auc': []
}

for model_name, report in models_data.items():
    performance_table['models'].append(model_name)
    performance_table['accuracy'].append(round(report['accuracy'], 3))
    performance_table['precision'].append(round(report['weighted avg']['precision'], 3))
    performance_table['recall'].append(round(report['weighted avg']['recall'], 3))
    performance_table['f1_score'].append(round(report['weighted avg']['f1-score'], 3))
    
    if model_name == 'SigLIP':
        auc = roc_auc_score(y_true, siglip_probs)
    elif model_name == 'ViT':
        auc = roc_auc_score(y_true, vit_probs)
    else:
        auc = roc_auc_score(y_true, ensemble_probs)
    performance_table['auc'].append(round(auc, 3))

with open(os.path.join(output_folder, 'performance_comparison.json'), 'w') as f:
    json.dump(performance_table, f, indent=4)

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(performance_table['models']))
width = 0.15

ax.bar(x - 2*width, performance_table['accuracy'], width, label='Accuracy', color='#3498db')
ax.bar(x - width, performance_table['precision'], width, label='Precision', color='#2ecc71')
ax.bar(x, performance_table['recall'], width, label='Recall', color='#e74c3c')
ax.bar(x + width, performance_table['f1_score'], width, label='F1-Score', color='#f39c12')
ax.bar(x + 2*width, performance_table['auc'], width, label='AUC', color='#9b59b6')

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: SigLIP vs ViT (Gemini) vs Ensemble', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(performance_table['models'])
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'performance_comparison.png'), dpi=150)
plt.close()
print("✓ Performance comparison saved")

# ============ REST OF THE CODE (classwise, dataset distribution, ROC curves, etc.) ============
# ... (Keep all the other code from previous evaluate_model.py) ...

# ============ SUMMARY ============
summary = {
    'total_images': len(y_true),
    'fracture_images': len(fracture_images),
    'no_fracture_images': len(no_fracture_images),
    'models_used': ['SigLIP', 'ViT (Real Gemini)', 'Ensemble (Average)'],
    'siglip_accuracy': round(accuracy_score(y_true, siglip_pred), 3),
    'vit_accuracy': round(accuracy_score(y_true, vit_pred), 3),
    'ensemble_accuracy': round(accuracy_score(y_true, ensemble_pred), 3),
    'ensemble_auc': round(auc, 3)
}

with open(os.path.join(output_folder, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n{'='*60}")
print("EVALUATION COMPLETE WITH REAL GEMINI!")
print(f"{'='*60}")
print(f"\nModel Accuracies:")
print(f"  SigLIP: {summary['siglip_accuracy']*100:.2f}%")
print(f"  ViT (Gemini): {summary['vit_accuracy']*100:.2f}%")
print(f"  Ensemble: {summary['ensemble_accuracy']*100:.2f}%")
print(f"\n{'='*60}")
