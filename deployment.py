!pip install streamlit
!pip install opencv-python
!pip install ultralytics



import streamlit as st
import cv2
import numpy as np
import time
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
from io import BytesIO
import os

# Import your YOLO model and set up video paths here
from ultralytics import YOLO
model = YOLO('C:/Users/Dell Latitude 5490/Downloads/best.pt')
# Function to process video
def process_video(uploaded_file, model, names):
    track_history = defaultdict(lambda: [])

    # Save uploaded file to disk
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    output_video_path = os.path.join("D:\\YOLOv8.1\\experiment_1.1\\weights", "annotated_video.mp4")
    result = cv2.VideoWriter(output_video_path,
                           cv2.VideoWriter_fourcc(*'mp4v'),
                           fps,
                           (w, h))

    start_time = time.time()  # Record start time

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:

                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, track_id in zip(boxes, clss, track_ids):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)

                    # Plot tracks
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                    cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            result.write(frame)
        else:
             break

    result.release()
    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time

    # Remove temporary file
    os.remove(temp_file_path)

    return execution_time


# Streamlit UI
def main():
    st.title("Video Annotation and Object Tracking")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        model = YOLO(r"C:/Users/Dell Latitude 5490/Downloads/best.pt")

        # Define names variable here or import it from your_yolo_module
        names = model.model.names

        execution_time = process_video(uploaded_file, model, names)
        st.write("Processing time:", execution_time, "seconds")

if __name__ == "__main__":
    main()
