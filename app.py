import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLO11 Object Detection", page_icon="🎯", layout="wide")

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Settings")
model_type = st.sidebar.selectbox("Select Model Size", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
model = load_model(model_type)

# --- MAIN UI ---
st.title("🎯 YOLO11 Object Detection")
st.write("Upload an image to see the model in action.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to PIL Image
    img = Image.open(uploaded_file)
    
    # Run Inference
    with st.spinner('Running detection...'):
        results = model.predict(img, conf=conf_threshold)
        
        # Plot the results on the image
        # result[0].plot() returns a BGR numpy array
        res_plotted = results[0].plot()[:, :, ::-1] # Convert BGR to RGB
        
    # Layout: Two columns for Original vs Detected
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)
        
    with col2:
        st.subheader("Detections")
        st.image(res_plotted, use_container_width=True)

    # --- DETECTION SUMMARY ---
    st.divider()
    st.subheader("📊 Detection Summary")
    
    # Get detection counts
    boxes = results[0].boxes
    if len(boxes) > 0:
        names = model.names
        detected_classes = [names[int(box.cls)] for box in boxes]
        counts = {name: detected_classes.count(name) for name in set(detected_classes)}
        
        # Display as metrics
        cols = st.columns(len(counts))
        for i, (name, count) in enumerate(counts.items()):
            cols[i].metric(label=name.upper(), value=count)
    else:
        st.info("No objects detected with the current confidence threshold.")

else:
    st.info("Please upload an image to start.")
