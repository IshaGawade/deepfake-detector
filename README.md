# Deepfake Detection with Explainable Visualization

Detects whether an **image or video** is **real or fake** using a Vision Transformer (ViT) model, with interactive visualizations and Grad-CAM for explainability.

## Models

- **dima806/deepfake_vs_real_image_detection** – Main ViT-based detection model
- **Xception (timm)** – Used only for generating Grad-CAM heatmaps

## Features

- Verdict card (Real/Fake + confidence)
- Probability gauge
- Frame-by-frame timeline *(video only)*
- Real vs Fake frame distribution *(video only)*
- Top 5 suspicious frames *(video only)*
- Grad-CAM attention heatmap

## Project Structure

```text
deepfake-detector/
├── utils/
│   ├── __init__.py
│   ├── model.py
│   ├── gradcam.py
│   ├── video.py
│   └── charts.py
├── app.py
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/IshaGawade/deepfake-detector.git
cd deepfake-detector

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- Streamlit
- PyTorch
- Transformers
- timm
- OpenCV
- Plotly
- Pillow

## Author

**Isha Gawade**
