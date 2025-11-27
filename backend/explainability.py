import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


def generate_gradcam_hf(model, inputs, original_image, model_name="Model"):
    """
    Generate Grad-CAM for HuggingFace transformer models
    """
    try:
        model.eval()
        
        # Enable gradients for input
        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                inputs[key].requires_grad = True
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get the predicted class score (fracture class - usually index 1)
        if logits.shape[1] > 1:
            target_score = logits[0, 1]
        else:
            target_score = logits[0, 0]
        
        # Backward pass
        model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Get gradients from the input
        pixel_values = inputs['pixel_values']
        gradients = pixel_values.grad
        
        if gradients is None:
            print(f"{model_name}: No gradients available, using fallback")
            return create_fallback_heatmap(original_image)
        
        # Create heatmap
        gradients = gradients.cpu().detach().numpy()[0]
        activation = pixel_values.cpu().detach().numpy()[0]
        
        # Weight the channels by their gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activation.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activation[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to original image size
        cam = cv2.resize(cam, original_image.size)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original
        original_np = np.array(original_image)
        if len(original_np.shape) == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        
        print(f"✓ {model_name} Grad-CAM generated")
        return overlay
        
    except Exception as e:
        print(f"{model_name} Grad-CAM error: {e}")
        return create_fallback_heatmap(original_image)


def create_fallback_heatmap(original_image):
    """
    Create a simple edge-based heatmap as fallback
    """
    try:
        img_gray = np.array(original_image.convert('L'))
        edges = cv2.Canny(img_gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        
        original_np = np.array(original_image)
        if len(original_np.shape) == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        
        overlay = cv2.addWeighted(original_np, 0.7, heatmap, 0.3, 0)
        return overlay
    except:
        return np.array(original_image)


def generate_lime_hf(original_image, model, processor):
    """
    LIME explanation for HuggingFace models
    """
    try:
        def batch_predict(images):
            batch_images = []
            for img in images:
                pil_img = Image.fromarray(img.astype('uint8'))
                batch_images.append(pil_img)
            
            inputs = processor(images=batch_images, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            return probs.cpu().numpy()
        
        explainer = lime_image.LimeImageExplainer()
        image_np = np.array(original_image.resize((224, 224)))
        
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        explanation = explainer.explain_instance(
            image_np,
            batch_predict,
            top_labels=2,
            hide_color=0,
            num_samples=50
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        lime_img = mark_boundaries(temp / 255.0, mask)
        lime_img = (lime_img * 255).astype(np.uint8)
        
        print("✓ LIME interpretation generated")
        return lime_img
        
    except Exception as e:
        print(f"LIME error: {e}")
        return np.array(original_image.resize((224, 224)))
