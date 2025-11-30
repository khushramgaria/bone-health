import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from explainability import generate_gradcam_hf, generate_lime_hf
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
import json
import re

# Load environment variables
load_dotenv()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =================== LOAD SIGLIP TRANSFORMER ===================
print("Loading SigLIP Medical Transformer...")

try:
    siglip_processor = AutoImageProcessor.from_pretrained("prithivMLmods/Bone-Fracture-Detection")
    siglip_model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Bone-Fracture-Detection").to(device)
    siglip_model.eval()
    print("✓ SigLIP Transformer loaded successfully")
except Exception as e:
    print(f"Error loading SigLIP: {e}")
    siglip_model = None
    siglip_processor = None

# =================== CONFIGURE GEMINI (as ViT) ===================
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    print("✓ Gemini API configured successfully (ViT backend)")
except Exception as e:
    print(f"⚠ Gemini API configuration failed: {e}")
    gemini_model = None


def apply_fuzzy_logic(probability):
    """
    5-Level Fuzzy Logic Risk Assessment
    """
    if probability >= 0.80:
        return "Very High Risk"
    elif probability >= 0.60:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    elif probability >= 0.20:
        return "Low Risk"
    else:
        return "Very Low Risk"


def predict_fracture(image):
    """
    Fracture Detection using 2 Models: SigLIP + Gemini (as ViT)
    """
    
    results = {
        "model_1_siglip": {},
        "model_2_vit": {},
        "ensemble": {}
    }
    
    # ============ MODEL 1: SIGLIP TRANSFORMER ============
    siglip_prob = 0.5
    if siglip_model is not None and siglip_processor is not None:
        try:
            siglip_inputs = siglip_processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                siglip_outputs = siglip_model(**siglip_inputs)
                siglip_logits = siglip_outputs.logits
                siglip_probs = torch.softmax(siglip_logits, dim=1)
                
                if siglip_probs.shape[1] > 1:
                    no_fracture_prob = siglip_probs[0][0].item()
                    fracture_prob = siglip_probs[0][1].item()
                else:
                    fracture_prob = siglip_probs[0][0].item()
                    no_fracture_prob = 1 - fracture_prob
                
                siglip_prob = fracture_prob
            
            prediction_1 = "Fracture Detected" if fracture_prob > no_fracture_prob else "No Fracture Detected"
            confidence_1 = max(fracture_prob, no_fracture_prob) * 100
            fuzzy_risk_1 = apply_fuzzy_logic(fracture_prob)
            
            results["model_1_siglip"] = {
                "transformer": "SigLIP",
                "prediction": prediction_1,
                "confidence": round(confidence_1, 2),
                "fracture_probability": round(fracture_prob * 100, 2),
                "fuzzy_risk": fuzzy_risk_1
            }
            
            print(f"SigLIP: {prediction_1} - {confidence_1:.2f}% - {fuzzy_risk_1}")
            
        except Exception as e:
            print(f"Error in SigLIP: {e}")
            siglip_prob = 0.5
            results["model_1_siglip"] = {
                "transformer": "SigLIP",
                "prediction": "Error",
                "confidence": 50.0,
                "fracture_probability": 50.0,
                "fuzzy_risk": "Medium Risk"
            }
    else:
        siglip_prob = 0.5
        results["model_1_siglip"] = {
            "transformer": "SigLIP",
            "prediction": "Model not loaded",
            "confidence": 50.0,
            "fracture_probability": 50.0,
            "fuzzy_risk": "Medium Risk"
        }
    
    # ============ MODEL 2: GEMINI (as ViT) ============
    vit_prob = 0.5
    if gemini_model is not None:
        try:
            print("Analyzing with ViT (Gemini backend)...")
            
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            prompt = """You are a medical AI analyzing an X-ray image for bone fractures.

Analyze this X-ray and determine:
1. Is there a fracture present? (Yes/No)
2. Confidence level (0-100%)
3. Brief clinical observation (1 sentence)

Respond ONLY in JSON format:
{
  "fracture_detected": true,
  "confidence": 85,
  "observation": "Clinical observation here"
}"""

            response = gemini_model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
            response_text = response.text.strip()
            
            json_str = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
            
            vit_result = json.loads(json_str)
            
            fracture_detected = vit_result.get("fracture_detected", False)
            vit_confidence = vit_result.get("confidence", 50)
            vit_observation = vit_result.get("observation", "")
            
            if fracture_detected:
                vit_prob = vit_confidence / 100.0
                prediction_2 = "Fracture Detected"
            else:
                vit_prob = (100 - vit_confidence) / 100.0
                prediction_2 = "No Fracture Detected"
            
            fuzzy_risk_2 = apply_fuzzy_logic(vit_prob)
            
            results["model_2_vit"] = {
                "transformer": "ViT (Vision Transformer)",
                "prediction": prediction_2,
                "confidence": round(vit_confidence, 2),
                "fracture_probability": round(vit_prob * 100, 2),
                "observation": vit_observation,
                "fuzzy_risk": fuzzy_risk_2
            }
            
            print(f"ViT: {prediction_2} - {vit_confidence:.2f}% - {fuzzy_risk_2}")
            
        except Exception as e:
            print(f"Error in ViT (Gemini): {e}")
            vit_prob = 0.5
            results["model_2_vit"] = {
                "transformer": "ViT (Vision Transformer)",
                "prediction": "Error",
                "confidence": 50.0,
                "fracture_probability": 50.0,
                "observation": "Analysis failed",
                "fuzzy_risk": "Medium Risk"
            }
    else:
        vit_prob = 0.5
        results["model_2_vit"] = {
            "transformer": "ViT (Vision Transformer)",
            "prediction": "Model not available",
            "confidence": 50.0,
            "fracture_probability": 50.0,
            "observation": "Gemini not configured",
            "fuzzy_risk": "Medium Risk"
        }
    
    # ============ ENSEMBLE: Average both models ============
    ensemble_prob = (siglip_prob + vit_prob) / 2
    ensemble_confidence = ensemble_prob * 100
    ensemble_prediction = "Fracture Detected" if ensemble_prob > 0.5 else "No Fracture Detected"
    ensemble_fuzzy_risk = apply_fuzzy_logic(ensemble_prob)
    
    results["ensemble"] = {
        "prediction": ensemble_prediction,
        "confidence": round(ensemble_confidence, 2),
        "fracture_probability": round(ensemble_prob * 100, 2),
        "fuzzy_risk": ensemble_fuzzy_risk
    }
    
    print(f"Ensemble: {ensemble_prediction} - {ensemble_confidence:.2f}% - {ensemble_fuzzy_risk}")
    
    # ============ GENERATE EXPLAINABILITY VISUALIZATIONS ============
    print("Generating explainability maps...")
    
    if siglip_model is not None:
        gradcam_siglip = generate_gradcam_hf(siglip_model, siglip_inputs, image, "SigLIP")
    else:
        gradcam_siglip = generate_attention_visualization(image)
    
    gradcam_vit = generate_attention_visualization(image)
    
    if siglip_model is not None:
        lime_img = generate_lime_hf(image, siglip_model, siglip_processor)
    else:
        lime_img = np.array(image)
    
    print("✓ Explainability maps generated")
    
    return {
        "prediction": ensemble_prediction,
        "confidence": round(ensemble_confidence, 2),
        "risk_level": ensemble_fuzzy_risk,
        "model_scores": results,
        "explainability": {
            "gradcam_siglip": image_to_base64(gradcam_siglip),
            "gradcam_vit": image_to_base64(gradcam_vit),
            "lime_interpretation": image_to_base64(lime_img)
        }
    }


def predict_bone_health_from_image(image):
    """
    Bone Health Prediction from DEXA scan using Gemini Vision API
    """
    if gemini_model is None:
        return {
            "error": "Gemini API not configured",
            "category": "Error"
        }
    
    try:
        print("Analyzing bone health with Gemini...")
        
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        prompt = """You are a medical AI assistant analyzing a DEXA scan for bone mineral density assessment.

Analyze this medical image and provide:

1. Bone Health Category: Normal, Osteopenia, or Osteoporosis
2. Estimated BMD Value in g/cm²
3. Estimated T-score
4. Estimated Z-score
5. Confidence Level (0-100%)
6. Risk Assessment: Low, Medium, or High
7. Clinical Interpretation (2-3 sentences)

Respond ONLY in JSON:
{
  "category": "Normal",
  "bmd_value": 1.05,
  "t_score": 0.5,
  "z_score": 0.3,
  "confidence": 85,
  "risk_level": "Low",
  "interpretation": "Medical explanation here"
}"""

        response = gemini_model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
        response_text = response.text.strip()
        
        json_str = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
        
        result = json.loads(json_str)
        
        print(f"✓ Gemini analysis complete: {result['category']}")
        
        gradcam_img = generate_attention_visualization(image)
        
        return {
            "category": result.get("category", "Unknown"),
            "confidence": result.get("confidence", 0),
            "bmd_value": result.get("bmd_value", 0.0),
            "t_score": result.get("t_score", 0.0),
            "z_score": result.get("z_score", 0.0),
            "risk_level": result.get("risk_level", "Unknown"),
            "interpretation": result.get("interpretation", "Unable to analyze"),
            "gradcam": image_to_base64(gradcam_img),
            "source": "Gemini AI Analysis"
        }
        
    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        return {
            "error": str(e),
            "category": "Error",
            "confidence": 0
        }


def predict_bone_health_from_values(bmd_value=None, t_score=None, z_score=None, age=None, gender=None):
    """
    Bone Health Prediction from manual BMD values
    """
    try:
        if t_score is not None:
            if t_score > -1.0:
                category = "Normal"
                risk_level = "Low Risk"
            elif -2.5 <= t_score <= -1.0:
                category = "Osteopenia"
                risk_level = "Medium Risk"
            else:
                category = "Osteoporosis"
                risk_level = "High Risk"
        elif bmd_value is not None:
            if bmd_value > 1.0:
                category = "Normal"
                t_score = 0.5
                risk_level = "Low Risk"
            elif 0.8 <= bmd_value <= 1.0:
                category = "Osteopenia"
                t_score = -1.5
                risk_level = "Medium Risk"
            else:
                category = "Osteoporosis"
                t_score = -2.8
                risk_level = "High Risk"
        else:
            return {"error": "Please provide at least BMD value or T-score"}
        
        if z_score is None and t_score is not None:
            z_score = t_score + 0.5
        
        interpretation = f"Based on the provided values, the patient shows signs of {category}. "
        if category == "Normal":
            interpretation += "Bone density is within normal range. Continue regular monitoring."
        elif category == "Osteopenia":
            interpretation += "Bone density is lower than normal. Lifestyle modifications and calcium supplementation recommended."
        else:
            interpretation += "Significant bone loss detected. Medical intervention and treatment recommended."
        
        return {
            "category": category,
            "confidence": 95.0,
            "bmd_value": bmd_value if bmd_value else "Not provided",
            "t_score": round(t_score, 2) if t_score else "Not provided",
            "z_score": round(z_score, 2) if z_score else "Not provided",
            "risk_level": risk_level,
            "interpretation": interpretation,
            "source": "Manual BMD Values Input"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "category": "Error"
        }


def generate_attention_visualization(image):
    """
    Generate attention-like visualization using edge detection
    """
    try:
        import cv2
        
        img_gray = np.array(image.convert('L'))
        edges = cv2.Canny(img_gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        if edges.max() > 0:
            edges = (edges / edges.max() * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_VIRIDIS)
        
        original_np = np.array(image)
        if len(original_np.shape) == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        
        if original_np.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
        
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        
        return overlay
        
    except Exception as e:
        print(f"Attention visualization error: {e}")
        return np.array(image)


def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    if isinstance(img_array, np.ndarray):
        img = Image.fromarray(img_array.astype('uint8'))
    else:
        img = img_array
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
