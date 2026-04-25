import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Video Frame Extractor
# ─────────────────────────────────────────────────────────────

def extract_frames(video_path, every_nth=15):
    """
    Extracts every Nth frame from a video file using OpenCV.

    Args:
        video_path : str — path to the temporary video file
        every_nth  : int — sample every Nth frame (default: 15)

    Returns:
        frames         : list of PIL.Image — sampled frames
        frame_numbers  : list of int       — original frame indices
        total_frames   : int               — total frame count in video
        fps            : float             — frames per second of video
    """
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_numbers = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_nth == 0:
            # Convert BGR (OpenCV) → RGB → PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)
            frames.append(pil_frame)
            frame_numbers.append(frame_idx)
        frame_idx += 1

    cap.release()
    return frames, frame_numbers, total_frames, fps
