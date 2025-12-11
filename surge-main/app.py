import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import time
import os
import tempfile

# --------------------------------------------------
# UI
# --------------------------------------------------
st.set_page_config(page_title="AI Needle Planner", layout="wide")

st.markdown("""
<style>
body { background-color: #111; }
h1, h2, h3 { color: #4FC3F7 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>AI Needle Path Planner (Video Simulation)</h1>",
            unsafe_allow_html=True)


# --------------------------------------------------
# Detect Yellow Tumor Box
# --------------------------------------------------
def detect_yellow_box(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 80, 80])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w//2, y + h//2

    return cx, cy, x, y, x+w, y+h


# --------------------------------------------------
# Hardcoded Smooth Path
# --------------------------------------------------
def generate_smooth_path(entry, target, steps=60):
    (x0, y0) = entry
    (x1, y1) = target

    path = []
    for t in np.linspace(0, 1, steps):
        xt = (1-t)**2 * x0 + 2*(1-t)*t*(x0 + (x1-x0)*0.3) + t**2 * x1
        yt = (1-t)**2 * y0 + 2*(1-t)*t*(y0 + (y1-y0)*0.15) + t**2 * y1
        path.append((int(xt), int(yt)))

    return path


# --------------------------------------------------
# Create Simulation Video (MP4)
# --------------------------------------------------
def create_simulation_video(base_img, path, fps=20):

    frames = []
    base_np = np.array(base_img)

    for (x, y) in path:
        frame = base_img.copy()
        draw = ImageDraw.Draw(frame)
        draw.ellipse([x-6, y-6, x+6, y+6], fill="cyan", outline="white", width=2)
        frames.append(np.array(frame))

    h, w, _ = frames[0].shape

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "needle_simulation.mp4")

    # OpenCV expects BGR, so convert frames
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    out.release()
    return video_path


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
uploaded = st.file_uploader("Upload CT Slice", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded CT", use_column_width=True)

    tumor = detect_yellow_box(img)
    if not tumor:
        st.error("Could not detect yellow tumor box. Use an image with a yellow highlighted box.")
        st.stop()

    cx, cy, x0, y0, x1, y1 = tumor

    annotated = img.copy()
    d = ImageDraw.Draw(annotated)
    d.rectangle([x0, y0, x1, y1], outline="yellow", width=4)
    d.ellipse([cx-6, cy-6, cx+6, cy+6], fill="red")

    st.image(annotated, caption="Detected Tumor Box", use_column_width=True)

    entry = (40, img.height // 2)
    target = (cx, cy)

    st.success(f"Entry point: {entry}")

    path = generate_smooth_path(entry, target)

    # Show preview image
    preview = annotated.copy()
    pdraw = ImageDraw.Draw(preview)
    for i in range(len(path)-1):
        pdraw.line([path[i], path[i+1]], fill="cyan", width=3)

    st.image(preview, caption="Planned Smooth Path", use_column_width=True)

    st.subheader("Generate Simulation Video")
    fps = st.slider("Video FPS", 5, 60, 24)

    if st.button("Create Needle Simulation Video"):
        video_file = create_simulation_video(preview, path, fps)
        st.video(video_file)

        with open(video_file, "rb") as f:
            st.download_button(
                "Download Simulation MP4",
                f,
                file_name="needle_simulation.mp4",
                mime="video/mp4"
            )

        st.success("Video generated successfully!")
