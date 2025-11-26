import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from explainability import generate_gradcam_simple, generate_lime, generate_attention_rollout
import base64
from io import BytesIO

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =================== LOAD HUGGING FACE PRETRAINED MODEL ===================
print("Loading Hugging Face bone fracture model...")
hf_processor = AutoImageProcessor.from_pretrained("prithivMLmods/Bone-Fracture-Detection")
hf_model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Bone-Fracture-Detection").to(device)
hf_model.eval()
print("✓ Hugging Face model loaded")

# =================== LOAD 3 TRANSFORMERS FOR EXPLAINABILITY ===================
print("Loading 3 transformer models (Swin, ViT, DeiT)...")
swin_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2).to(device)
vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=2).to(device)
deit_model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=2).to(device)

swin_model.eval()
vit_model.eval()
deit_model.eval()
print("✓ All 3 transformers loaded")

# Image preprocessing for timm models
preprocess_timm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_fracture(image):
    """
    Fracture Detection using Hugging Face model + 3 Transformers Ensemble
    """
    
    # ============ PRIMARY PREDICTION: Hugging Face Model ============
    hf_inputs = hf_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
        hf_logits = hf_outputs.logits
        hf_probs = torch.softmax(hf_logits, dim=1)
        hf_fracture_prob = hf_probs[0][1].item()  # Probability of fracture class
    
    # ============ ENSEMBLE: Run 3 Transformers ============
    img_tensor = preprocess_timm(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        swin_out = torch.softmax(swin_model(img_tensor), dim=1)
        vit_out = torch.softmax(vit_model(img_tensor), dim=1)
        deit_out = torch.softmax(deit_model(img_tensor), dim=1)
    
    # Get fracture probabilities from each model
    swin_prob = swin_out[0][1].item()
    vit_prob = vit_out[0][1].item()
    deit_prob = deit_out[0][1].item()
    
    # Weighted ensemble: Give more weight to Hugging Face model (trained on medical data)
    ensemble_prob = (
        hf_fracture_prob * 0.7 +  # 70% weight to HF model
        swin_prob * 0.1 +
        vit_prob * 0.1 +
        deit_prob * 0.1
    )
    
    confidence = ensemble_prob * 100
    prediction = "Fracture Detected" if ensemble_prob > 0.5 else "No Fracture Detected"
    
    # ============ FUZZY LOGIC RISK EVALUATION ============
    if confidence > 80:
        risk_level = "High Risk"
    elif confidence > 60:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    # ============ GENERATE EXPLAINABILITY VISUALIZATIONS ============
    print("Generating explainability maps...")
    
    # Grad-CAM from Swin Transformer
    gradcam_img = generate_gradcam_simple(swin_model, img_tensor, image)
    
    # LIME from ViT
    lime_img = generate_lime(image, vit_model, preprocess_timm)
    
    # Attention Rollout from DeiT
    attention_img = generate_attention_rollout(deit_model, img_tensor, image)
    
    print("✓ Explainability maps generated")
    
    # ============ RETURN RESULTS ============
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "risk_level": risk_level,
        "model_scores": {
            "huggingface_model": round(hf_fracture_prob * 100, 2),
            "swin_transformer": round(swin_prob * 100, 2),
            "vit_transformer": round(vit_prob * 100, 2),
            "deit_transformer": round(deit_prob * 100, 2),
            "ensemble": round(ensemble_prob * 100, 2)
        },
        "gradcam": image_to_base64(gradcam_img),
        "lime": image_to_base64(lime_img),
        "attention": image_to_base64(attention_img)
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
    img = Image.fromarray(img_array.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
