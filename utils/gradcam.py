import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import matplotlib.cm as cm
import torchvision.transforms as transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Preprocessing for Xception (299x299, normalized to [-1, 1])
# ─────────────────────────────────────────────────────────────

xception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ─────────────────────────────────────────────────────────────
# Load Xception model (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Grad-CAM model (Xception)...")
def load_xception():
    model = timm.create_model("xception", pretrained=True)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Grad-CAM computation
# ─────────────────────────────────────────────────────────────

def compute_gradcam(image_pil, xception_model):
    """
    Computes Grad-CAM heatmap and returns:
      - blended     : PIL.Image  — heatmap overlaid on original
      - cam_resized : np.ndarray — raw normalized CAM (H x W) for zone scoring
    """
    img_tensor = xception_transform(image_pil).unsqueeze(0)
    img_tensor.requires_grad_(True)

    activations = {}
    gradients = {}

    target_layer = xception_model.conv4

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    output = xception_model(img_tensor)
    xception_model.zero_grad()
    class_idx = output.argmax(dim=1).item()
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)

    fwd_handle.remove()
    bwd_handle.remove()

    act  = activations['value'].squeeze(0)
    grad = gradients['value'].squeeze(0)
    weights = grad.mean(dim=(1, 2))

    cam = torch.zeros(act.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    orig_w, orig_h = image_pil.size
    cam_np = cam.numpy()
    cam_resized = cv2.resize(cam_np, (orig_w, orig_h))

    colormap = cm.get_cmap('RdYlBu_r')
    heatmap_rgba = colormap(cam_resized)
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil  = Image.fromarray(heatmap_rgb)

    orig_rgb = image_pil.convert("RGB").resize((orig_w, orig_h))
    blended  = Image.blend(orig_rgb, heatmap_pil, alpha=0.5)

    return blended, cam_resized


# ─────────────────────────────────────────────────────────────
# Facial Zone Scoring
# ─────────────────────────────────────────────────────────────

def compute_zone_scores(cam_resized):
    """
    Divides the CAM into 6 approximate facial regions and returns
    the mean activation % for each zone.

    Zones are based on rough proportional splits of a face image:
      Forehead  — top 20%
      Left Eye  — top 20–40%, left half
      Right Eye — top 20–40%, right half
      Nose      — middle 40–60%
      Lips      — 60–80%
      Jaw       — bottom 20%

    Args:
        cam_resized : np.ndarray — normalized CAM (H x W), values 0–1

    Returns:
        dict of zone_name -> activation % (0–100)
    """
    h, w = cam_resized.shape

    zones = {
        "Forehead":  cam_resized[0          : int(h*0.20), :],
        "Left Eye":  cam_resized[int(h*0.20): int(h*0.40), :w//2],
        "Right Eye": cam_resized[int(h*0.20): int(h*0.40), w//2:],
        "Nose":      cam_resized[int(h*0.40): int(h*0.60), :],
        "Lips":      cam_resized[int(h*0.60): int(h*0.80), :],
        "Jaw":       cam_resized[int(h*0.80): h,           :],
    }

    scores = {}
    for name, region in zones.items():
        scores[name] = float(np.mean(region) * 100)

    return scores
