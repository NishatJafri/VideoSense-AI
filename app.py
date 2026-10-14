
import streamlit as st
import pandas as pd
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(
    page_title="VideoSense AI",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 VideoSense AI")
st.subheader("Intelligent Video Analysis Dashboard")
st.write("Explore object detections and confidence scores from your video.")
uploaded_video = st.file_uploader(
    "Upload a video for analysis",
    type=["mp4", "avi", "mov", "mkv"]
)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()


if uploaded_video is not None:
    st.video(uploaded_video)

    if "video_detections" not in st.session_state:
        st.info("Click Analyze Video to generate results.")

    if st.button("🚀 Analyze Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.getvalue())
            video_path = temp_file.name

        detections = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        cap.release()

        with st.spinner("Analyzing video..."):
            results = model.predict(
                source=video_path,
                conf=0.25,
                stream=True,
                verbose=False
            )

            for frame_number, result in enumerate(results):
                for box in result.boxes:
                    detections.append({
                        "timestamp_seconds": frame_number / fps,
                        "frame": frame_number,
                        "object": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0])
                    })

        df = pd.DataFrame(
            detections,
            columns=[
                "timestamp_seconds", "frame",
                "object", "confidence"
            ]
        )

        st.session_state["video_detections"] = df

        st.success("Video analysis completed!")

if uploaded_video is not None:
    if "video_detections" not in st.session_state:
        st.stop()
    df = st.session_state["video_detections"]
else:
    df = pd.read_csv("/content/videosense_detections.csv")



col1, col2, col3 = st.columns(3)

col1.metric("Total Detections", len(df))
col2.metric("Unique Object Classes", df["object"].nunique())
col3.metric(
    "Average Confidence",
    f"{df['confidence'].mean():.1%}"
)

st.subheader("Detection Records")
st.dataframe(df, use_container_width=True)

st.subheader("Detections by Object Class")
object_counts = df["object"].value_counts()
st.bar_chart(object_counts)


st.subheader("📈 Frame-by-Frame Detection Trend")

frame_summary = (
    df.groupby(["timestamp_seconds", "frame"])
    .size()
    .reset_index(name="detection_count")
    .sort_values("timestamp_seconds")
)

st.line_chart(
    frame_summary,
    x="timestamp_seconds",
    y="detection_count"
)

st.caption(
    "Shows the number of object detections in each sampled video frame."
)


st.subheader("🎯 Detection Confidence Distribution")

st.bar_chart(
    df["confidence"].round(1).value_counts().sort_index()
)

st.caption(
    "Distribution of model confidence scores. "
    "Confidence is not the same as detection accuracy."
)


st.subheader("🔎 Filter Detections by Object")

object_options = sorted(df["object"].unique())

selected_objects = st.multiselect(
    "Choose object classes",
    options=object_options,
    default=object_options
)

filtered_df = df[df["object"].isin(selected_objects)]

st.write(f"Showing {len(filtered_df)} detections")
st.dataframe(filtered_df, use_container_width=True)



