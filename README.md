Deepfake Detection with Explainable Visualization
Detects whether an image or video is real or fake using a Vision Transformer model, with interactive visualizations and a Grad-CAM heatmap for explainability.
---
Models
dima806/deepfake_vs_real_image_detection — main detection model (ViT)
Xception via timm — used only for Grad-CAM heatmap
---
Visualizations
Verdict card (Real/Fake + confidence)
Probability gauge
Frame-by-frame timeline — video only
Real vs Fake frame bar chart — video only
Top 5 suspicious frames grid — video only
Grad-CAM attention heatmap
---
Project Structure
```
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
---
Setup
```bash
git clone https://github.com/IshaGawade/deepfake-detector.git
cd deepfake-detector
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
---
Tech Stack
Streamlit, Plotly, OpenCV, Transformers, timm, PyTorch, Pillow
---
Author
Isha Gawade