import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoProcessor, AutoModel
from explainability import generate_gradcam_hf, generate_lime_hf
import base64
from io import BytesIO

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =================== LOAD 2 HUGGINGFACE MEDICAL TRANSFORMERS ===================

print("Loading HuggingFace Medical Models...")

# Model 1: SigLIP-based Bone Fracture Detection
try:
    hf_processor_1 = AutoImageProcessor.from_pretrained("prithivMLmods/Bone-Fracture-Detection")
    hf_model_1 = AutoModelForImageClassification.from_pretrained("prithivMLmods/Bone-Fracture-Detection").to(device)
    hf_model_1.eval()
    print("✓ Model 1 (SigLIP Transformer) loaded")
except Exception as e:
    print(f"Error loading Model 1: {e}")
    hf_model_1 = None

# Model 2: ViT-based Bone Fracture Detection
try:
    hf_processor_2 = AutoImageProcessor.from_pretrained("Hemgg/bone-fracture-detection-using-xray")
    hf_model_2 = AutoModelForImageClassification.from_pretrained("Hemgg/bone-fracture-detection-using-xray").to(device)
    hf_model_2.eval()
    print("✓ Model 2 (ViT Transformer) loaded")
except Exception as e:
    print(f"Error loading Model 2: {e}")
    hf_model_2 = None

print("All models loaded successfully!")


def predict_fracture(image):
    """
    Fracture Detection using 2 HuggingFace Medical Transformers
    """
    
    results = {
        "model_1_siglip": {},
        "model_2_vit": {},
        "ensemble": {}
    }
    
    # ============ MODEL 1: SigLIP TRANSFORMER ============
    if hf_model_1 is not None:
        try:
            hf_inputs_1 = hf_processor_1(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                hf_outputs_1 = hf_model_1(**hf_inputs_1)
                hf_logits_1 = hf_outputs_1.logits
                hf_probs_1 = torch.softmax(hf_logits_1, dim=1)
                
                # Get fracture probability (assuming class 1 is fracture)
                fracture_prob_1 = hf_probs_1[0][1].item() if hf_probs_1.shape[1] > 1 else hf_probs_1[0][0].item()
                prediction_1 = "Fracture Detected" if fracture_prob_1 > 0.5 else "No Fracture Detected"
            
            results["model_1_siglip"] = {
                "transformer": "SigLIP",
                "prediction": prediction_1,
                "confidence": round(fracture_prob_1 * 100, 2)
            }
            
            print(f"Model 1 (SigLIP): {prediction_1} - {fracture_prob_1*100:.2f}%")
            
        except Exception as e:
            print(f"Error in Model 1: {e}")
            fracture_prob_1 = 0.5
            results["model_1_siglip"] = {
                "transformer": "SigLIP",
                "prediction": "Error",
                "confidence": 50.0
            }
    else:
        fracture_prob_1 = 0.5
    
    # ============ MODEL 2: ViT TRANSFORMER ============
    if hf_model_2 is not None:
        try:
            hf_inputs_2 = hf_processor_2(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                hf_outputs_2 = hf_model_2(**hf_inputs_2)
                hf_logits_2 = hf_outputs_2.logits
                hf_probs_2 = torch.softmax(hf_logits_2, dim=1)
                
                fracture_prob_2 = hf_probs_2[0][1].item() if hf_probs_2.shape[1] > 1 else hf_probs_2[0][0].item()
                prediction_2 = "Fracture Detected" if fracture_prob_2 > 0.5 else "No Fracture Detected"
            
            results["model_2_vit"] = {
                "transformer": "ViT (Vision Transformer)",
                "prediction": prediction_2,
                "confidence": round(fracture_prob_2 * 100, 2)
            }
            
            print(f"Model 2 (ViT): {prediction_2} - {fracture_prob_2*100:.2f}%")
            
        except Exception as e:
            print(f"Error in Model 2: {e}")
            fracture_prob_2 = 0.5
            results["model_2_vit"] = {
                "transformer": "ViT",
                "prediction": "Error",
                "confidence": 50.0
            }
    else:
        fracture_prob_2 = 0.5
    
    # ============ ENSEMBLE: Average both models ============
    ensemble_prob = (fracture_prob_1 + fracture_prob_2) / 2
    confidence = ensemble_prob * 100
    prediction = "Fracture Detected" if ensemble_prob > 0.5 else "No Fracture Detected"
    
    results["ensemble"] = {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
    
    print(f"Ensemble: {prediction} - {confidence:.2f}%")
    
    # ============ FUZZY LOGIC RISK EVALUATION ============
    if confidence > 80:
        risk_level = "High Risk"
    elif confidence > 60:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    # ============ GENERATE EXPLAINABILITY VISUALIZATIONS ============
    print("Generating explainability maps...")
    
    # Grad-CAM from Model 1 (SigLIP)
    if hf_model_1 is not None:
        gradcam_img_1 = generate_gradcam_hf(hf_model_1, hf_inputs_1, image, "Model 1: SigLIP")
    else:
        gradcam_img_1 = np.array(image)
    
    # Grad-CAM from Model 2 (ViT)
    if hf_model_2 is not None:
        gradcam_img_2 = generate_gradcam_hf(hf_model_2, hf_inputs_2, image, "Model 2: ViT")
    else:
        gradcam_img_2 = np.array(image)
    
    # LIME from Model 1
    if hf_model_1 is not None:
        lime_img = generate_lime_hf(image, hf_model_1, hf_processor_1)
    else:
        lime_img = np.array(image)
    
    print("✓ Explainability maps generated")
    
    # ============ RETURN RESULTS ============
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "risk_level": risk_level,
        "model_scores": {
            "model_1_siglip": results["model_1_siglip"],
            "model_2_vit": results["model_2_vit"],
            "ensemble": round(ensemble_prob * 100, 2)
        },
        "explainability": {
            "gradcam_siglip": image_to_base64(gradcam_img_1),
            "gradcam_vit": image_to_base64(gradcam_img_2),
            "lime_interpretation": image_to_base64(lime_img)
        }
    }


def predict_bone_health(image):
    """
    Placeholder for Module 1 - We'll implement this later
    """
    return {
        "message": "Module 1 (Bone Health) - Coming soon",
        "category": "Normal",
        "confidence": 0.0
    }


def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    if isinstance(img_array, np.ndarray):
        img = Image.fromarray(img_array.astype('uint8'))
    else:
        img = img_array
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
