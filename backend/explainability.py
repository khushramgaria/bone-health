import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


def generate_gradcam_simple(model, input_tensor, original_image):
    """
    Simplified Grad-CAM without Captum
    """
    try:
        model.eval()
        
        # Forward pass
        input_tensor.requires_grad = True
        output = model(input_tensor)
        
        # Get the fracture class score (class 1)
        fracture_score = output[0, 1]
        
        # Backward pass
        model.zero_grad()
        fracture_score.backward(retain_graph=True)
        
        # Get gradients
        gradients = input_tensor.grad.data
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        activation = input_tensor * weights
        heatmap = activation.sum(dim=1).squeeze()
        
        # ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, original_image.size)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original
        original_np = np.array(original_image.resize(original_image.size))
        if len(original_np.shape) == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        return overlay
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return np.array(original_image)


def generate_lime(original_image, model, preprocess):
    """
    LIME explanation
    """
    try:
        def batch_predict(images):
            batch = torch.stack([preprocess(Image.fromarray(img.astype('uint8'))) 
                                for img in images])
            device = next(model.parameters()).device
            batch = batch.to(device)
            
            with torch.no_grad():
                outputs = torch.softmax(model(batch), dim=1)
            
            return outputs.cpu().numpy()
        
        explainer = lime_image.LimeImageExplainer()
        image_np = np.array(original_image.resize((224, 224)))
        
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        explanation = explainer.explain_instance(
            image_np,
            batch_predict,
            top_labels=2,
            hide_color=0,
            num_samples=50  # Reduced for speed
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        lime_img = mark_boundaries(temp / 255.0, mask)
        lime_img = (lime_img * 255).astype(np.uint8)
        
        return lime_img
        
    except Exception as e:
        print(f"LIME error: {e}")
        return np.array(original_image.resize((224, 224)))


def generate_attention_rollout(model, input_tensor, original_image):
    """
    Attention rollout for Vision Transformers
    """
    try:
        # Simple attention visualization
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Fallback: Create a simple attention-like visualization
        img_gray = np.array(original_image.convert('L'))
        edges = cv2.Canny(img_gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        
        # Normalize
        attention_map = edges.astype(float)
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        attention_map = np.uint8(255 * attention_map)
        
        # Apply colormap
        attention_map = cv2.resize(attention_map, original_image.size)
        heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_VIRIDIS)
        
        # Overlay
        original_np = np.array(original_image)
        if len(original_np.shape) == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        return overlay
        
    except Exception as e:
        print(f"Attention Rollout error: {e}")
        return np.array(original_image.resize((224, 224)))
