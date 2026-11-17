import hashlib
import io

import pandas as pd
import streamlit as st
from PIL import Image

from src.image_inference import (
    annotate_image, calculate_speedup, create_parallel_engine,
    detect_and_crop_faces, run_parallel, run_sequential,
)


st.set_page_config(
    page_title="Parallel Emotion Detection System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #f5f8fc; color: #1c2b3a; font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 3.5rem; }
h1, h2, h3 { color: #102a43; font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; }
h2 { font-size: 1.38rem !important; margin: 0.2rem 0 0.75rem; }
p, [data-testid="stCaptionContainer"] { color: #5c6f82; }

.hero { background: linear-gradient(115deg, #ffffff 0%, #f1f7ff 100%); border: 1px solid #dce8f5; border-radius: 18px; padding: 2rem 2.1rem; box-shadow: 0 10px 28px rgba(28, 75, 122, 0.07); margin-bottom: 1.65rem; }
.eyebrow { color: #2573b9; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.55rem; }
.hero-title { color: #102a43; font-family: 'Manrope', sans-serif; font-size: clamp(1.8rem, 3vw, 2.55rem); font-weight: 800; letter-spacing: -0.05em; line-height: 1.14; }
.hero-subtitle { color: #61758a; font-size: 1rem; margin-top: 0.65rem; }

[data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; border: 1px solid #dfe8f2; border-radius: 15px; box-shadow: 0 6px 20px rgba(29, 67, 105, 0.045); }
[data-testid="stMetric"] { background: linear-gradient(145deg, #ffffff, #f7fbff); border: 1px solid #d9e6f3; border-radius: 13px; padding: 1.05rem 1.1rem; box-shadow: 0 5px 16px rgba(24, 62, 101, 0.055); transition: transform .2s ease, box-shadow .2s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(24, 62, 101, 0.1); }
[data-testid="stMetricLabel"] { color: #62758a; font-size: 0.8rem; font-weight: 600; }
[data-testid="stMetricValue"] { color: #145f9e; font-family: 'Manrope', sans-serif; font-weight: 700; }

.stButton > button { border: 1px solid #cbddec; border-radius: 9px; color: #24577e; font-weight: 700; min-height: 2.8rem; padding: 0.45rem 1rem; transition: all .2s ease; }
.stButton > button:hover { border-color: #2b78ba; color: #175b94; transform: translateY(-1px); box-shadow: 0 7px 16px rgba(36, 112, 177, .14); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #176eae, #2582c3); border-color: #176eae; color: white; box-shadow: 0 7px 16px rgba(31, 112, 178, .2); }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #105d98, #1874b5); color: white; }

[data-testid="stRadio"] { background: #edf4fb; border: 1px solid #d9e6f2; border-radius: 10px; padding: 0.28rem 0.6rem; width: fit-content; }
[data-testid="stRadio"] label { color: #526d85; font-weight: 600; }
[data-testid="stFileUploader"] { background: #fbfdff; border: 1px dashed #a9c8e2; border-radius: 13px; padding: 0.55rem; transition: border-color .2s ease, background .2s ease; }
[data-testid="stFileUploader"]:hover { background: #f3f9ff; border-color: #4b91c7; }
[data-testid="stCameraInput"] { border: 1px solid #d9e6f2; border-radius: 13px; padding: 0.55rem; background: #fbfdff; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #d8e4ef; box-shadow: 0 8px 22px rgba(26, 67, 108, .09); }
div[data-testid="stDataFrame"] { border: 1px solid #dce7f1; border-radius: 10px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 10px; }
hr { border-color: #e1eaf2; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


def clear_analysis():
    """Remove a completed analysis while retaining the chosen input image."""
    for key in (
        "output_image", "face_count", "sequential_time", "parallel_time",
        "speedup", "parallel_results", "analysis_image_id",
    ):
        st.session_state.pop(key, None)


def load_image(uploaded_file):
    """Return a durable RGB image and an ID based on its source bytes."""
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image, hashlib.sha256(image_bytes).hexdigest()


st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered analysis</div>
  <div class="hero-title">Parallel Emotion Detection System</div>
  <div class="hero-subtitle">Facial Emotion Recognition using Convolutional Neural Networks and Multiprocessing</div>
</div>
""", unsafe_allow_html=True)

st.subheader("Image Input")
st.caption("Select an image from your webcam or upload one from your device.")
input_method = st.radio(
    "Input method", ("Capture Image", "Upload Image"), horizontal=True,
    label_visibility="collapsed",
)

selected_image = None
selected_image_id = None
if input_method == "Capture Image":
    camera_image = st.camera_input("Capture an image using your webcam", key="camera_image")
    if camera_image is not None:
        selected_image, selected_image_id = load_image(camera_image)
else:
    uploaded_image = st.file_uploader(
        "Upload a JPG or PNG image", type=["jpg", "jpeg", "png"], key="image_upload",
    )
    if uploaded_image is not None:
        selected_image, selected_image_id = load_image(uploaded_image)

if selected_image is None:
    st.info("Choose a capture or upload an image to begin analysis.")
else:
    input_column, action_column = st.columns((2, 1), gap="large")
    with input_column:
        st.image(selected_image, caption="Selected input image", use_container_width=True)
    with action_column:
        st.markdown("#### Ready to analyze")
        st.write("The system will detect faces, classify each expression, and compare measured sequential and parallel execution times.")
        analyze_clicked = st.button("Analyze Image", type="primary", use_container_width=True)

        if analyze_clicked:
            try:
                with st.spinner("Detecting faces and analyzing emotions..."):
                    image_bgr, faces, crops = detect_and_crop_faces(selected_image)
                    if not crops:
                        clear_analysis()
                        st.warning("No faces were detected. Please select another image.")
                    else:
                        sequential_results, sequential_time = run_sequential(crops)
                        if (
                            "parallel_engine" not in st.session_state
                            or st.session_state["parallel_engine"].closed
                        ):
                            st.session_state["parallel_engine"] = create_parallel_engine(workers=4)
                        parallel_results, parallel_time = run_parallel(
                            crops, st.session_state["parallel_engine"]
                        )
                        st.session_state.update({
                            "output_image": annotate_image(image_bgr, faces, parallel_results),
                            "face_count": len(faces),
                            "sequential_time": sequential_time,
                            "parallel_time": parallel_time,
                            "speedup": calculate_speedup(sequential_time, parallel_time),
                            "parallel_results": parallel_results,
                            "analysis_image_id": selected_image_id,
                        })
                        st.success("Analysis complete.")
            except Exception as error:
                st.error(f"Unable to process this image: {error}")


has_current_analysis = (
    "output_image" in st.session_state
    and selected_image_id == st.session_state.get("analysis_image_id")
)
if has_current_analysis:
    st.divider()
    st.subheader("Emotion Detection Output")
    result_column, details_column = st.columns((2, 1), gap="large")
    with result_column:
        st.image(st.session_state["output_image"], caption="Detected faces and predicted emotions", use_container_width=True)
    with details_column:
        st.metric("Detected faces", st.session_state["face_count"])
        st.markdown("#### Individual Predictions")
        prediction_data = [{
            "Person ID": f"Person {result['person_id']}",
            "Predicted Emotion": result["emotion"],
            "Confidence": f"{result['confidence']:.2f}%",
        } for result in st.session_state["parallel_results"]]
        st.dataframe(pd.DataFrame(prediction_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Sequential vs Parallel Performance")
    sequential_ms = st.session_state["sequential_time"] * 1000
    parallel_ms = st.session_state["parallel_time"] * 1000
    speedup = st.session_state["speedup"]
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Sequential Execution Time", f"{sequential_ms:.2f} ms")
    metric_two.metric("Parallel Execution Time", f"{parallel_ms:.2f} ms")
    metric_three.metric("Speedup", f"{speedup:.2f}×")

    comparison_data = pd.DataFrame({
        "Method": ["Sequential", "Parallel"],
        "Execution time (ms)": [sequential_ms, parallel_ms],
    }).set_index("Method")
    st.bar_chart(comparison_data, color="#2675b9", use_container_width=True)

    if speedup > 1:
        st.success(f"Parallel execution was {speedup:.2f}× faster for this workload.")
    elif speedup < 1:
        st.info("Sequential execution was faster for this workload because multiprocessing overhead dominated.")
    else:
        st.info("Both approaches took approximately the same time for this workload.")

    if st.button("Analyze Another Image"):
        clear_analysis()
        st.rerun()
elif "output_image" in st.session_state and selected_image is not None:
    st.info("This is a new input image. Select Analyze Image to generate its results.")

st.divider()
st.caption("CNN · OpenCV · Python Multiprocessing")
