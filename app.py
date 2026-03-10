import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Detection App")

# Question 1 - Instructions
st.markdown("""
## 📋 How to Use This App
1. **Upload an image** using the button below (JPG or PNG)
2. **Adjust settings** in the sidebar
3. Click **Detect Faces** button
4. **Download** the result using the button below

> ℹ️ This app uses OpenCV's Haar Cascade classifier for frontal face detection.
""")

# Sidebar settings
st.sidebar.title("⚙️ Settings")

# Question 3 - Color picker
hex_color = st.sidebar.color_picker("🎨 Rectangle color", "#00FF00")
hex_stripped = hex_color.lstrip("#")
r = int(hex_stripped[0:2], 16)
g = int(hex_stripped[2:4], 16)
b = int(hex_stripped[4:6], 16)
bgr_color = (b, g, r)

# Question 4 - minNeighbors slider
min_neighbors = st.sidebar.slider(
    "minNeighbors",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Higher = stricter detection. Lower = more detections."
)

# Question 5 - scaleFactor slider
scale_factor = st.sidebar.slider(
    "scaleFactor",
    min_value=1.01,
    max_value=2.0,
    value=1.1,
    step=0.01,
    help="Lower = more thorough but slower. Higher = faster but may miss faces."
)

# Question 2 - Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original image")

    if st.button("🔍 Detect Faces"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Question 4 + 5 - both sliders used here
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), bgr_color, 3)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"{len(faces)} face(s) detected — scaleFactor={scale_factor} | minNeighbors={min_neighbors}"
        )

        # Question 2 - Download
        result = cv2.imencode(".png", img)[1].tobytes()
        st.download_button(
            label="💾 Download image with detected faces",
            data=result,
            file_name="detected_faces.png",
            mime="image/png"
        )
