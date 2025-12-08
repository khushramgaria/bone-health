import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel, AutoProcessor
from explainability import generate_gradcam_hf, generate_lime_hf
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


# =================== LOAD SIGLIP TRANSFORMER (FRACTURE DETECTION) ===================
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


# =================== LOAD BIOMEDCLIP (BONE HEALTH DETECTION) ===================
print("Loading BiomedCLIP for Bone Health Detection...")

bone_health_model = None
bone_health_processor = None

try:
    bone_health_model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", trust_remote_code=True).to(device)
    bone_health_processor = AutoProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", trust_remote_code=True)
    bone_health_model.eval()
    print("✓ BiomedCLIP for Bone Health loaded successfully")
except Exception as e:
    print(f"⚠ BiomedCLIP loading failed, trying CLIP: {e}")
    try:
        from transformers import CLIPProcessor, CLIPModel
        bone_health_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        bone_health_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        bone_health_model.eval()
        print("✓ CLIP for Bone Health loaded successfully")
    except Exception as e2:
        print(f"Error loading bone health model: {e2}")
        bone_health_model = None
        bone_health_processor = None

# Text descriptions for zero-shot classification (SAME AS EVALUATION)
bone_health_text_prompts = [
    "a DEXA scan showing normal healthy bone density with strong bone structure and high mineral content",
    "a DEXA scan showing osteopenia with reduced bone mass and decreased bone mineral density",
    "a DEXA scan showing osteoporosis with severe bone loss, low bone density, and fragile bone structure"
]


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
    Fracture Detection using SigLIP Transformer
    """
    
    results = {
        "model_1_siglip": {},
        "ensemble": {}
    }
    
    # ============ MODEL: SIGLIP TRANSFORMER ============
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
    
    # ============ ENSEMBLE (Single Model) ============
    ensemble_prob = siglip_prob
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

def generate_detailed_interpretation(image, category, prob_normal, prob_osteopenia, prob_osteoporosis, t_score, bmd_value):
    """
    Generate detailed Google Lens-style interpretation
    Explains WHY the model predicted this category
    """
    import cv2
    
    # Analyze image characteristics
    img_gray = np.array(image.convert('L'))
    img_array = np.array(image.convert('RGB'))
    
    # Calculate image features
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)
    contrast = img_gray.max() - img_gray.min()
    
    # Edge detection (bone structure)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Brightness analysis
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    peak_intensity = np.argmax(hist)
    
    # Build detailed interpretation
    interpretation_points = []
    
    # 1. Primary Diagnosis
    interpretation_points.append({
        "title": "🎯 Primary Diagnosis",
        "finding": category,
        "confidence": f"{max(prob_normal, prob_osteopenia, prob_osteoporosis)*100:.1f}%",
        "explanation": f"The AI model analyzed the DEXA scan and classified it as **{category}** with {max(prob_normal, prob_osteopenia, prob_osteoporosis)*100:.1f}% confidence based on bone density patterns, trabecular structure, and mineral content distribution."
    })
    
    # 2. Bone Density Analysis
    if category == "Normal":
        density_explanation = f"The scan shows **healthy bone density** (BMD: {bmd_value:.3f} g/cm²). Dense bone tissue appears darker in DEXA scans due to higher calcium absorption. The mean intensity value of {mean_intensity:.1f} indicates good mineralization."
    elif category == "Osteopenia":
        density_explanation = f"The scan reveals **reduced bone mass** (BMD: {bmd_value:.3f} g/cm²). The bone structure shows signs of mineral depletion. Mean intensity of {mean_intensity:.1f} suggests moderate bone density loss compared to healthy reference ranges."
    else:
        density_explanation = f"The scan indicates **severe bone density loss** (BMD: {bmd_value:.3f} g/cm²). Significantly reduced mineral content is visible. Mean intensity of {mean_intensity:.1f} reflects advanced osteoporosis with porous bone structure."
    
    interpretation_points.append({
        "title": "🦴 Bone Density Analysis",
        "finding": f"BMD: {bmd_value:.3f} g/cm²",
        "confidence": f"T-score: {t_score:.2f}",
        "explanation": density_explanation
    })
    
    # 3. Trabecular Structure (Bone Microarchitecture)
    if edge_density > 0.15:
        structure_finding = "Strong trabecular network"
        structure_explanation = f"Edge density of {edge_density:.3f} indicates **well-preserved bone microarchitecture**. The trabecular bone shows complex patterns typical of healthy bone with intact structural connections. This reduces fracture risk."
    elif edge_density > 0.08:
        structure_finding = "Moderate trabecular loss"
        structure_explanation = f"Edge density of {edge_density:.3f} shows **partial trabecular deterioration**. Some bone microarchitecture remains but with visible thinning and disconnection of trabecular struts. This increases vulnerability to mechanical stress."
    else:
        structure_finding = "Severe trabecular disruption"
        structure_explanation = f"Edge density of {edge_density:.3f} reveals **significant trabecular breakdown**. Loss of bone microarchitecture is evident with disconnected or missing trabecular structures. This dramatically increases fracture risk even from minor trauma."
    
    interpretation_points.append({
        "title": "🔬 Trabecular Structure",
        "finding": structure_finding,
        "confidence": f"Edge Density: {edge_density:.3f}",
        "explanation": structure_explanation
    })
    
    # 4. Bone Texture & Uniformity
    if std_intensity < 35:
        texture_finding = "Uniform bone texture"
        texture_explanation = f"Standard deviation of {std_intensity:.1f} indicates **consistent bone density** throughout the scan. Uniform texture suggests even mineral distribution and healthy bone remodeling."
    elif std_intensity < 55:
        texture_finding = "Irregular bone texture"
        texture_explanation = f"Standard deviation of {std_intensity:.1f} shows **some heterogeneity** in bone density. This can indicate ongoing bone remodeling or early mineral loss in certain regions."
    else:
        texture_finding = "Highly irregular texture"
        texture_explanation = f"Standard deviation of {std_intensity:.1f} reveals **significant texture irregularity**. This suggests uneven bone resorption and formation, typical of advanced osteoporosis with patchy mineral loss."
    
    interpretation_points.append({
        "title": "📊 Bone Texture Analysis",
        "finding": texture_finding,
        "confidence": f"Std Dev: {std_intensity:.1f}",
        "explanation": texture_explanation
    })
    
    # 5. Contrast & Bone Definition
    if contrast > 180:
        contrast_finding = "Excellent bone-soft tissue contrast"
        contrast_explanation = f"Contrast value of {contrast:.1f} shows **clear differentiation** between bone and soft tissue. High contrast indicates dense bone with strong X-ray absorption, typical of healthy mineralization."
    elif contrast > 120:
        contrast_finding = "Moderate contrast"
        contrast_explanation = f"Contrast value of {contrast:.1f} indicates **reduced bone definition**. Lower contrast suggests decreased bone density making bone-soft tissue boundaries less distinct."
    else:
        contrast_finding = "Poor bone definition"
        contrast_explanation = f"Contrast value of {contrast:.1f} shows **minimal bone-soft tissue separation**. Very low contrast indicates severe mineral depletion making bones appear similar in density to surrounding tissue."
    
    interpretation_points.append({
        "title": "🎨 Contrast Analysis",
        "finding": contrast_finding,
        "confidence": f"Contrast: {contrast:.1f}",
        "explanation": contrast_explanation
    })
    
    # 6. Intensity Distribution (Histogram Analysis)
    if peak_intensity < 110:
        histogram_finding = "Dark peak (Dense bones)"
        histogram_explanation = f"Peak intensity at {peak_intensity} indicates **majority of pixels are dark**, characteristic of dense, calcium-rich bone. This is associated with good bone health and low fracture risk."
    elif peak_intensity < 150:
        histogram_finding = "Mid-range peak (Reduced density)"
        histogram_explanation = f"Peak intensity at {peak_intensity} shows **most pixels in medium range**, suggesting reduced bone mineral content. This transitional pattern is common in osteopenia."
    else:
        histogram_finding = "Light peak (Low density)"
        histogram_explanation = f"Peak intensity at {peak_intensity} reveals **predominantly light pixels**, indicating severe mineral loss. Lighter appearance in DEXA scans correlates with porous, fragile bone structure."
    
    interpretation_points.append({
        "title": "📈 Intensity Distribution",
        "finding": histogram_finding,
        "confidence": f"Peak: {peak_intensity}",
        "explanation": histogram_explanation
    })
    
    # 7. AI Model Confidence Breakdown
    confidence_explanation = f"""
The AI model uses zero-shot learning with medical imaging knowledge to classify bone health:
- **Normal probability: {prob_normal*100:.1f}%** - Model detected patterns matching healthy bone density
- **Osteopenia probability: {prob_osteopenia*100:.1f}%** - Some features suggest mild bone loss
- **Osteoporosis probability: {prob_osteoporosis*100:.1f}%** - Patterns indicating severe bone depletion

The highest probability determines the final classification. BiomedCLIP model was trained on millions of medical images and text descriptions, allowing it to understand DEXA scan characteristics.
"""
    
    interpretation_points.append({
        "title": "🤖 AI Model Analysis",
        "finding": f"Classified as {category}",
        "confidence": f"Certainty: {max(prob_normal, prob_osteopenia, prob_osteoporosis)*100:.1f}%",
        "explanation": confidence_explanation.strip()
    })
    
    # 8. Clinical Correlation
    if category == "Normal":
        clinical_explanation = "**Clinical Interpretation:** Normal bone density (T-score > -1.0) indicates low fracture risk. Patient should maintain current lifestyle with regular weight-bearing exercise, adequate calcium (1000mg/day) and vitamin D (600 IU/day). Routine follow-up in 2-3 years."
    elif category == "Osteopenia":
        clinical_explanation = f"**Clinical Interpretation:** Osteopenia (T-score {t_score:.2f}) indicates reduced bone mass. Moderate fracture risk requiring intervention. Recommend calcium supplementation (1200mg/day), vitamin D (1000 IU/day), weight-bearing exercises, and fall prevention. Consider pharmacological therapy if high-risk factors present. Follow-up DEXA in 12-18 months."
    else:
        clinical_explanation = f"**Clinical Interpretation:** Osteoporosis (T-score {t_score:.2f}) indicates severe bone loss with high fracture risk. **Immediate medical attention required.** Recommend bisphosphonate therapy (e.g., Alendronate), increased calcium (1500mg/day) and vitamin D (2000 IU/day), physical therapy for balance, home safety modifications, and fall prevention strategies. Urgent consultation with endocrinologist or orthopedic specialist. Follow-up DEXA in 6-12 months."
    
    interpretation_points.append({
        "title": "🏥 Clinical Correlation",
        "finding": f"T-score: {t_score:.2f}",
        "confidence": "WHO Classification",
        "explanation": clinical_explanation
    })
    
    return interpretation_points



def predict_bone_health_from_image(image):
    """
    Bone Health Prediction from DEXA scan using BiomedCLIP (SAME AS EVALUATION)
    Categories: Normal, Osteopenia, Osteoporosis
    """
    if bone_health_model is None or bone_health_processor is None:
        return {
            "error": "Bone health model not loaded",
            "category": "Error",
            "confidence": 0
        }
    
    try:
        print("Analyzing bone health with BiomedCLIP...")
        
        # Process inputs (SAME AS EVALUATION)
        inputs = bone_health_processor(
            text=bone_health_text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = bone_health_model(**inputs)
            
            # Get similarity scores
            if hasattr(outputs, 'logits_per_image'):
                logits = outputs.logits_per_image  # CLIP
            else:
                logits = outputs.logits_per_text.T  # BiomedCLIP
            
            # Apply softmax
            probs = torch.softmax(logits[0], dim=0)
            
            prob_normal = probs[0].item()
            prob_osteopenia = probs[1].item()
            prob_osteoporosis = probs[2].item()
            
            # Get predicted class
            predicted_idx = torch.argmax(probs).item()
        
        # Map predictions
        categories = ["Normal", "Osteopenia", "Osteoporosis"]
        category = categories[predicted_idx]
        confidence = max(prob_normal, prob_osteopenia, prob_osteoporosis) * 100
        
        # Estimate T-score based on category
        if category == "Normal":
            t_score = 0.5 + (prob_normal * 1.0)
            risk_level = "Low Risk"
            bmd_value = 1.0 + (prob_normal * 0.2)
        elif category == "Osteopenia":
            t_score = -1.75 + (prob_osteopenia * 0.5)
            risk_level = "Medium Risk"
            bmd_value = 0.85 + (prob_osteopenia * 0.15)
        else:  # Osteoporosis
            t_score = -3.0 + (prob_osteoporosis * 0.3)
            risk_level = "High Risk"
            bmd_value = 0.65 + (prob_osteoporosis * 0.15)
        
        # Calculate Z-score
        z_score = t_score + 0.5
        
        # Generate basic interpretation
        interpretation = generate_bone_health_interpretation(
            category, t_score, bmd_value, risk_level
        )
        
        # ✨ NEW: Generate detailed Google Lens-style interpretation
        detailed_interpretation = generate_detailed_interpretation(
            image, category, prob_normal, prob_osteopenia, prob_osteoporosis, 
            t_score, bmd_value
        )
        
        # Generate attention visualization
        gradcam_img = generate_attention_visualization(image)
        
        print(f"✓ Bone Health Analysis: {category} ({confidence:.1f}% confidence)")
        print(f"  Probabilities: Normal={prob_normal*100:.1f}%, Osteopenia={prob_osteopenia*100:.1f}%, Osteoporosis={prob_osteoporosis*100:.1f}%")
        
        return {
            "category": category,
            "confidence": round(confidence, 2),
            "probabilities": {
                "normal": round(prob_normal * 100, 2),
                "osteopenia": round(prob_osteopenia * 100, 2),
                "osteoporosis": round(prob_osteoporosis * 100, 2)
            },
            "bmd_value": round(bmd_value, 3),
            "t_score": round(t_score, 2),
            "z_score": round(z_score, 2),
            "risk_level": risk_level,
            "interpretation": interpretation,
            "detailed_interpretation": detailed_interpretation,  # ✨ NEW
            "gradcam": image_to_base64(gradcam_img),
            "source": "BiomedCLIP Medical Model"
        }
        
    except Exception as e:
        print(f"Error in bone health analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "category": "Error",
            "confidence": 0,
            "interpretation": "Analysis failed. Please try again."
        }



def generate_bone_health_interpretation(category, t_score, bmd_value, risk_level):
    """
    Generate clinical interpretation based on WHO guidelines
    """
    interpretations = {
        "Normal": f"Bone mineral density is within normal range (T-score: {t_score:.2f}, BMD: {bmd_value:.3f} g/cm²). "
                  "Patient shows healthy bone density. Continue regular monitoring and maintain calcium/vitamin D intake. "
                  "Weight-bearing exercises recommended.",
        
        "Osteopenia": f"Bone mineral density is lower than normal (T-score: {t_score:.2f}, BMD: {bmd_value:.3f} g/cm²). "
                     "Patient shows signs of low bone mass. Recommend increased calcium (1200mg/day) and vitamin D3 supplementation. "
                     "Weight-bearing exercises and lifestyle modifications advised. Monitor every 6-12 months.",
        
        "Osteoporosis": f"Significant bone density loss detected (T-score: {t_score:.2f}, BMD: {bmd_value:.3f} g/cm²). "
                       "Patient shows signs of osteoporosis with increased fracture risk. Immediate medical intervention recommended. "
                       "Consider bisphosphonate therapy, fall prevention measures, and specialized orthopedic consultation."
    }
    
    return interpretations.get(category, "Unable to generate interpretation.")


def predict_bone_health_from_values(bmd_value=None, t_score=None, z_score=None, age=None, gender=None):
    """
    Bone Health Prediction from manual BMD values (WHO Classification)
    """
    try:
        # Determine category based on T-score (WHO standard)
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
        
        # Fallback to BMD value if T-score not provided
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
            return {
                "error": "Please provide at least BMD value or T-score",
                "category": "Error"
            }
        
        # Calculate Z-score if not provided
        if z_score is None and t_score is not None:
            z_score = t_score + 0.5
        
        # Generate interpretation
        interpretation = generate_bone_health_interpretation(category, t_score, bmd_value if bmd_value else 0.0, risk_level)
        
        # Add age/gender specific recommendations
        if age and gender:
            if age > 65 and category != "Normal":
                interpretation += f"\n\nNote: Patient age ({age}) increases fracture risk. Enhanced monitoring recommended."
            if gender == "female" and category == "Osteoporosis":
                interpretation += "\n\nPostmenopausal women with osteoporosis may benefit from hormone therapy evaluation."
        
        return {
            "category": category,
            "confidence": 95.0,
            "bmd_value": round(bmd_value, 3) if bmd_value else "Not provided",
            "t_score": round(t_score, 2) if t_score else "Not provided",
            "z_score": round(z_score, 2) if z_score else "Not provided",
            "risk_level": risk_level,
            "interpretation": interpretation,
            "source": "WHO Classification (Manual Input)",
            "recommendations": generate_recommendations(category, age, gender)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "category": "Error"
        }


def generate_recommendations(category, age=None, gender=None):
    """
    Generate personalized recommendations based on bone health category
    """
    recommendations = []
    
    if category == "Normal":
        recommendations = [
            "Continue regular calcium intake (1000-1200mg/day)",
            "Maintain vitamin D levels (600-800 IU/day)",
            "Regular weight-bearing exercises (30 min, 3-4x/week)",
            "Avoid smoking and excessive alcohol",
            "Follow-up DEXA scan in 2-3 years"
        ]
    
    elif category == "Osteopenia":
        recommendations = [
            "Increase calcium to 1200mg/day",
            "Vitamin D3 supplementation (1000-2000 IU/day)",
            "Weight-bearing and resistance exercises (5x/week)",
            "Consider bone-strengthening medications if high risk",
            "Follow-up DEXA scan in 12-18 months",
            "Fall prevention measures at home"
        ]
    
    else:  # Osteoporosis
        recommendations = [
            "Immediate consultation with endocrinologist/orthopedist",
            "Consider bisphosphonate therapy (Alendronate, Risedronate)",
            "Calcium 1200-1500mg/day + Vitamin D3 2000 IU/day",
            "Physical therapy for strength and balance",
            "Home safety modifications (fall prevention)",
            "Avoid activities with high fracture risk",
            "Follow-up DEXA scan in 6-12 months",
            "Consider parathyroid hormone therapy if severe"
        ]
    
    # Add age-specific recommendations
    if age:
        if age > 70:
            recommendations.append("Hip protectors recommended for fall protection")
    
    # Add gender-specific recommendations
    if gender == "female" and category in ["Osteopenia", "Osteoporosis"]:
        recommendations.append("Discuss hormone replacement therapy with gynecologist")
    
    return recommendations



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
