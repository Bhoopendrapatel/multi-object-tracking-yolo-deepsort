import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.title("Multi-Object Detection & Tracking")

# Option select
option = st.radio("Choose Input Source", ["Upload Video", "Live Camera"])

model = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=800,
    n_init=1,
    max_cosine_distance=0.15,
    nn_budget=100
)

stframe = st.empty()

# ================== VIDEO UPLOAD ==================
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        # Output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output.mp4", fourcc, 20.0,
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            detections = []

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, 'person'))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())

                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            #  Save video
            out.write(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        cap.release()
        out.release()

        st.success(" Video Saved as output.mp4")


# ================== LIVE CAMERA ==================
elif option == "Live Camera":
    run = st.checkbox("Start Camera")

    cap = cv2.VideoCapture(0)

    # Output save for live cam
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("live_output.mp4", fourcc, 20.0,
                          (640, 480))

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save live video
        out.write(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
    out.release()

    if not run:
        st.warning("Camera stopped")
        st.success("Live video saved as live_output.mp4")
