import streamlit as st
from transformers import pipeline

# ─────────────────────────────────────────────────────────────
# Load dima806 ViT deepfake detection model (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading detection model...")
def load_detector():
    """
    Loads the dima806/deepfake_vs_real_image_detection pipeline.
    Cached so it loads only once per session.
    """
    return pipeline(
        "image-classification",
        model="dima806/deepfake_vs_real_image_detection"
    )


def predict(image_pil, detector):
    """
    Run the dima806 model on a PIL image.

    Args:
        image_pil : PIL.Image  — the image to analyze
        detector  : pipeline   — loaded HuggingFace pipeline

    Returns:
        label      : str   — "Fake" or "Real"
        fake_score : float — probability of being fake (0.0 to 1.0)
    """
    results = detector(image_pil)
    top = results[0]
    label = top['label']
    score = top['score']
    fake_score = score if label.lower() == "fake" else 1 - score
    return label, fake_score
