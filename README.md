# VideoSense-AI
AI-powered video analysis dashboard using YOLO, OpenCV, Python and Streamlit

# 🎥 VideoSense AI

**An AI-powered video analysis dashboard built with Python, YOLO, OpenCV, and Streamlit.**

VideoSense AI analyzes uploaded videos using a pretrained YOLO11n object detection model and presents the results through an interactive dashboard.

## ✨ Features

- **Video Upload:** Upload videos in MP4, AVI, MOV, and MKV formats.
- **Object Detection:** Detect objects using the YOLO11n pretrained model.
- **Interactive Dashboard:** View total detections, unique object classes, and average confidence.
- **Detection Records:** Explore timestamps, frame numbers, object classes, and confidence scores.
- **Visual Analytics:** View object-class distribution, frame-by-frame detection trends, and confidence-score distribution.
- **Object Filtering:** Filter detection records by object class.

## 🛠️ Tech Stack

- Python
- YOLO11n — Ultralytics
- OpenCV
- Streamlit
- Pandas

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/NishatJafri/VideoSense-AI.git
cd VideoSense-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🔍 How It Works

1. Upload a supported video.
2. Click **Analyze Video**.
3. YOLO processes the video frames and detects objects.
4. The application displays detection records, KPIs, and visualizations.
5. Use the object-class filter to explore specific detections.

## ⚠️ Limitations

- Detection quality depends on the model, video quality, and selected confidence threshold.
- Confidence scores are not equivalent to detection accuracy.
- Detected objects are counted across frames; they are not uniquely tracked identities.
- Processing time depends on video length, resolution, and available hardware.

## 🚀 Future Improvements

- Configurable frame sampling for faster analysis.
- Export filtered detection results.
- Display annotated video frames.
- Add further evaluation and performance metrics.

## 👩‍💻 Author

**Nishat Jafri**

[GitHub Profile](https://github.com/NishatJafri)

