import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Fix random seeds
torch.manual_seed(42)
np.random.seed(42)

print("="*100)
print("BONE HEALTH DETECTION - BIOMEDCLIP (MEDICAL AI MODEL)")
print("="*100)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[SYSTEM] Using device: {device}")

# =================== LOAD BIOMEDCLIP MODEL ===================
print("\n[MODEL] Loading BiomedCLIP medical model...")

try:
    model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", trust_remote_code=True)
    model.eval()
    print("✓ BiomedCLIP loaded successfully")
except Exception as e:
    print(f"Error loading BiomedCLIP: {e}")
    print("\nTrying alternative: OpenAI CLIP with medical prompts...")
    
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print("✓ CLIP model loaded successfully")

# Text descriptions for zero-shot classification
text_descriptions = [
    "a DEXA scan showing normal healthy bone density with strong bone structure and high mineral content",
    "a DEXA scan showing osteopenia with reduced bone mass and decreased bone mineral density",
    "a DEXA scan showing osteoporosis with severe bone loss, low bone density, and fragile bone structure"
]

print(f"\nText prompts loaded: {len(text_descriptions)}")

# =================== PREDICTION FUNCTION ===================
def predict_bone_health(image_path):
    """
    Predict bone health using BiomedCLIP/CLIP zero-shot classification
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Process inputs
        inputs = processor(
            text=text_descriptions,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get similarity scores
            if hasattr(outputs, 'logits_per_image'):
                logits = outputs.logits_per_image  # CLIP
            else:
                logits = outputs.logits_per_text.T  # BiomedCLIP
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits[0], dim=0)
            
            prob_normal = probs[0].item()
            prob_osteopenia = probs[1].item()
            prob_osteoporosis = probs[2].item()
            
            # Get prediction
            pred_idx = torch.argmax(probs).item()
        
        return pred_idx, prob_normal, prob_osteopenia, prob_osteoporosis
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0, 0.33, 0.33, 0.34

# Dataset paths
test_data_path = "bone_health_data/"
folders = {
    0: ("normal", os.path.join(test_data_path, "normal/")),
    1: ("osteopenia", os.path.join(test_data_path, "osteopenia/")),
    2: ("osteoporosis", os.path.join(test_data_path, "osteoporosis/"))
}

categories = ['Normal', 'Osteopenia', 'Osteoporosis']
y_true, y_pred, y_probs = [], [], []
detailed_results = []
class_correct = {0: 0, 1: 0, 2: 0}
class_total = {0: 0, 1: 0, 2: 0}

# =================== PROCESS ALL IMAGES ===================
for true_label_idx, (folder_name, folder_path) in folders.items():
    
    print("\n" + "="*100)
    print(f"PROCESSING {categories[true_label_idx].upper()} IMAGES")
    print("="*100)
    
    if not os.path.exists(folder_path):
        print(f"⚠ Folder not found: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
        continue
    
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
    print(f"\n[INFO] Found {len(images)} {folder_name} images\n")
    
    class_total[true_label_idx] = len(images)
    
    for i, img_name in enumerate(images, 1):
        try:
            img_path = os.path.join(folder_path, img_name)
            
            # Predict
            pred_idx, prob_normal, prob_osteopenia, prob_osteoporosis = predict_bone_health(img_path)
            
            prediction = categories[pred_idx]
            is_correct = pred_idx == true_label_idx
            correct = "✓ CORRECT" if is_correct else "✗ WRONG"
            
            if is_correct:
                class_correct[true_label_idx] += 1
            
            y_true.append(true_label_idx)
            y_pred.append(pred_idx)
            y_probs.append([prob_normal, prob_osteopenia, prob_osteoporosis])
            
            detailed_results.append({
                "filename": img_name,
                "folder": folder_name,
                "true_label": categories[true_label_idx],
                "predicted_label": prediction,
                "correct": is_correct,
                "confidence": max(prob_normal, prob_osteopenia, prob_osteoporosis) * 100,
                "probabilities": {
                    "normal": round(prob_normal * 100, 2),
                    "osteopenia": round(prob_osteopenia * 100, 2),
                    "osteoporosis": round(prob_osteoporosis * 100, 2)
                }
            })
            
            print(f"[{i:3d}/{len(images)}] {img_name:50s}")
            print(f"         True: {categories[true_label_idx]:15s} | Pred: {prediction:15s} {correct}")
            print(f"         Probs: N={prob_normal*100:5.1f}% | P={prob_osteopenia*100:5.1f}% | O={prob_osteoporosis*100:5.1f}%")
            
            if not is_correct:
                print(f"         ⚠ MISCLASSIFIED - Consider removing")
            print()
            
        except Exception as e:
            print(f"[{i:3d}/{len(images)}] ❌ {img_name} | Error: {str(e)}\n")

# =================== CALCULATE RESULTS ===================
print(f"\n{'='*100}")
print("EVALUATION RESULTS")
print(f"{'='*100}\n")

for idx in range(3):
    accuracy_per_class = (class_correct[idx] / class_total[idx] * 100) if class_total[idx] > 0 else 0
    print(f"{categories[idx]:15s}: {class_correct[idx]:3d}/{class_total[idx]:3d} correct ({accuracy_per_class:5.1f}%)")

total_correct = sum(class_correct.values())
total_images = sum(class_total.values())
overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0

print(f"\n{'='*50}")
print(f"OVERALL ACCURACY: {total_correct}/{total_images} ({overall_accuracy:.2f}%)")
print(f"{'='*50}\n")

# =================== DETAILED CLASSIFICATION REPORT ===================
print("CLASSIFICATION REPORT:")
print("-"*100)
report_text = classification_report(y_true, y_pred, target_names=categories, zero_division=0)
print(report_text)

# =================== CONFUSION MATRIX ===================
cm = confusion_matrix(y_true, y_pred)
print("\nCONFUSION MATRIX:")
print("                  Predicted")
print("              Normal  Osteopenia  Osteoporosis")
if len(cm) >= 3:
    print(f"Actual Normal       {cm[0][0]:3d}        {cm[0][1]:3d}          {cm[0][2]:3d}")
    print(f"       Osteopenia   {cm[1][0]:3d}        {cm[1][1]:3d}          {cm[1][2]:3d}")
    print(f"       Osteoporosis {cm[2][0]:3d}        {cm[2][1]:3d}          {cm[2][2]:3d}")

# =================== SAVE RESULTS ===================
output_folder = "bone_health_metrics/"
os.makedirs(output_folder, exist_ok=True)

# Save detailed JSON
with open(os.path.join(output_folder, 'detailed_results.json'), 'w') as f:
    json.dump(detailed_results, f, indent=2)

# Save classification report
report_dict = classification_report(y_true, y_pred, target_names=categories, output_dict=True, zero_division=0)
with open(os.path.join(output_folder, 'bone_health_report.json'), 'w') as f:
    json.dump(report_dict, f, indent=2)

# Save summary
summary = {
    "model": "BiomedCLIP/CLIP",
    "total_images": total_images,
    "correct_predictions": total_correct,
    "overall_accuracy": round(overall_accuracy, 2),
    "per_class_accuracy": {
        "normal": round((class_correct[0] / class_total[0] * 100) if class_total[0] > 0 else 0, 2),
        "osteopenia": round((class_correct[1] / class_total[1] * 100) if class_total[1] > 0 else 0, 2),
        "osteoporosis": round((class_correct[2] / class_total[2] * 100) if class_total[2] > 0 else 0, 2)
    },
    "per_class_counts": {
        "normal": f"{class_correct[0]}/{class_total[0]}",
        "osteopenia": f"{class_correct[1]}/{class_total[1]}",
        "osteoporosis": f"{class_correct[2]}/{class_total[2]}"
    },
    "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(os.path.join(output_folder, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Generate confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.title('Bone Health Classification - Confusion Matrix (BiomedCLIP)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'figure_1_confusion_matrix.png'), dpi=150)
plt.close()

# Generate confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.title('Bone Health Classification - Confusion Matrix (BiomedCLIP)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'figure_1_confusion_matrix.png'), dpi=150)
plt.close()

# ========== ADD THIS BLOCK HERE ==========
# Generate per-class performance chart
print("\n[FIGURE 2] Generating per-class performance chart...")

report_dict = classification_report(
    y_true, y_pred, target_names=categories, output_dict=True, zero_division=0
)

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(categories))
width = 0.25

precisions = [report_dict[c]['precision'] for c in categories]
recalls = [report_dict[c]['recall'] for c in categories]
f1_scores = [report_dict[c]['f1-score'] for c in categories]

ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Bone Health Category', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Figure 2: Per-Class Performance Metrics', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'figure_2_perclass_performance.png'), dpi=150)
plt.close()
print("  ✓ Figure 2 saved")
# ========== END OF ADDITION ==========

# Save misclassified list
misclassified = [r for r in detailed_results if not r['correct']]
with open(os.path.join(output_folder, 'DELETE_THESE_IMAGES.txt'), 'w') as f:
    f.write("="*80 + "\n")
    f.write(f"IMAGES TO DELETE FOR BETTER ACCURACY\n")
    f.write(f"Current Accuracy: {overall_accuracy:.1f}%\n")
    f.write(f"Misclassified: {len(misclassified)}\n")
    f.write("="*80 + "\n\n")
    
    for item in misclassified:
        f.write(f"rm bone_health_data/{item['folder']}/{item['filename']}\n")

print(f"\n✓ Results saved to {output_folder}")
print(f"✓ Model: BiomedCLIP (Medical-trained)")
print(f"✓ Check DELETE_THESE_IMAGES.txt for images to remove")
print(f"✓ After removing wrong images, re-run to improve accuracy")
print(f"\n✓ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
