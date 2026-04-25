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
    """
    Loads pretrained Xception from timm.
    Used ONLY for Grad-CAM heatmap — not for detection.
    Cached so it loads only once per session.
    """
    model = timm.create_model("xception", pretrained=True)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Grad-CAM computation
# ─────────────────────────────────────────────────────────────

def compute_gradcam(image_pil, xception_model):
    """
    Computes a Grad-CAM heatmap using Xception's last conv layer (conv4).
    Blends the heatmap with the original image for visualization.

    Args:
        image_pil      : PIL.Image  — input image
        xception_model : nn.Module  — loaded Xception model

    Returns:
        blended : PIL.Image — original image with heatmap overlay (50/50 blend)
    """
    img_tensor = xception_transform(image_pil).unsqueeze(0)
    img_tensor.requires_grad_(True)

    # Storage for forward/backward hooks
    activations = {}
    gradients = {}

    # Target: Xception's last conv block (conv4)
    target_layer = xception_model.conv4

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # ── Forward pass ──
    output = xception_model(img_tensor)

    # ── Backward on top predicted class ──
    xception_model.zero_grad()
    class_idx = output.argmax(dim=1).item()
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)

    fwd_handle.remove()
    bwd_handle.remove()

    # ── Compute weighted activation map ──
    act = activations['value'].squeeze(0)   # [C, H, W]
    grad = gradients['value'].squeeze(0)    # [C, H, W]
    weights = grad.mean(dim=(1, 2))         # [C] — global average pooling

    cam = torch.zeros(act.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    # ── Apply ReLU and normalize ──
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    # ── Resize to original image dimensions ──
    orig_w, orig_h = image_pil.size
    cam_np = cam.numpy()
    cam_resized = cv2.resize(cam_np, (orig_w, orig_h))

    # ── Apply colormap (red = high attention, blue = low) ──
    colormap = cm.get_cmap('RdYlBu_r')
    heatmap_rgba = colormap(cam_resized)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # ── Blend 50% original + 50% heatmap ──
    orig_rgb = image_pil.convert("RGB").resize((orig_w, orig_h))
    blended = Image.blend(orig_rgb, heatmap_pil, alpha=0.5)

    return blended
