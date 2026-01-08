import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLO11 Vision App", page_icon="🎯", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }

/* Summary section background */
[data-testid="stMetric"] {
    background-color: #1e1e1e !important;
    color: white !important;
    border-radius: 10px;
    padding: 15px;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #bbbbbb !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")

task_type = st.sidebar.selectbox(
    "Select Task",
    ["Object Detection", "Pose Estimation", "Image Classification"]
)

model_map = {
    "Object Detection": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
    "Pose Estimation": ["yolo11n-pose.pt"],
    "Image Classification": ["yolo11n-cls.pt"]
}

model_type = st.sidebar.selectbox("Select Model", model_map[task_type])

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

model = load_model(model_type)

# --- MAIN UI ---
st.title("🎯 YOLO11 Vision App")
st.write("Upload an image and run Detection, Pose, or Classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    with st.spinner("Running inference..."):
        results = model.predict(img, conf=conf_threshold)
        plotted = results[0].plot()
        plotted = plotted[:, :, ::-1] if isinstance(plotted, np.ndarray) else None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Result")
        if plotted is not None:
            st.image(plotted, use_container_width=True)
        else:
            st.info("No visual output for this task.")

    # --- SUMMARY ---
    st.divider()
    st.subheader("📊 Summary")

    res = results[0]

    # OBJECT DETECTION
    if task_type == "Object Detection":
        if res.boxes is not None and len(res.boxes) > 0:
            names = model.names
            classes = [names[int(b.cls)] for b in res.boxes]
            counts = {c: classes.count(c) for c in set(classes)}

            cols = st.columns(len(counts))
            for i, (k, v) in enumerate(counts.items()):
                cols[i].metric(k.upper(), v)
        else:
            st.info("No objects detected.")

    # POSE ESTIMATION
    elif task_type == "Pose Estimation":
        if res.keypoints is not None:
            st.metric("People Detected", len(res.keypoints))
            st.write("Pose keypoints detected for each person.")
        else:
            st.info("No poses detected.")

    # CLASSIFICATION
    elif task_type == "Image Classification":
        probs = res.probs
        top_class = model.names[int(probs.top1)]
        confidence = probs.top1conf.item()

        col1, col2 = st.columns(2)
        col1.metric("Predicted Class", top_class.upper())
        col2.metric("Confidence", f"{confidence:.2f}")

else:
    st.info("Please upload an image to start.")
