import cv2
import tempfile
import streamlit as st
import pandas as pd
from ultralytics import YOLO


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Speed Estimation",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Speed Estimation & Overspeed Detection")
st.write(
    "Detect vehicles, track them, estimate their speed, "
    "and identify overspeeding vehicles."
)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.1
)

known_distance = st.sidebar.slider(
    "Distance Between Lines (metres)",
    5,
    50,
    10,
    1
)

speed_limit = st.sidebar.slider(
    "Speed Limit (km/h)",
    10,
    150,
    40,
    5
)

st.sidebar.info(
    "Speed is estimated using the time taken by a "
    "tracked vehicle to travel between two lines."
)


# --------------------------------------------------
# VEHICLE CLASSES
# --------------------------------------------------

# COCO class IDs
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]

vehicle_names = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

uploaded_video = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)


# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------

if uploaded_video is not None:

    # Save uploaded video temporarily
    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(uploaded_video.read())
    temp_video.close()

    video_path = temp_video.name

    # Open video
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        st.error("Could not read video FPS.")
        st.stop()

    st.info(
        f"Video: {width} × {height} | "
        f"FPS: {fps:.2f} | "
        f"Frames: {total_frames}"
    )

    # --------------------------------------------------
    # TWO COUNTING LINES
    # --------------------------------------------------

    line_1 = int(height * 0.35)
    line_2 = int(height * 0.65)

    st.write(f"📏 Distance between lines: **{known_distance} metres**")
    st.write(f"🚦 Speed limit: **{speed_limit} km/h**")

    frame_placeholder = st.empty()

    progress_bar = st.progress(0)

    # --------------------------------------------------
    # TRACKING VARIABLES
    # --------------------------------------------------

    previous_positions = {}

    # Vehicle ID -> time when it crossed Line 1
    line1_times = {}

    # Vehicle ID -> final result
    speed_results = {}

    # Vehicle ID -> vehicle type
    vehicle_types = {}

    # Vehicle ID -> time taken
    time_results = {}

    frame_number = 0

    # --------------------------------------------------
    # PROCESS VIDEO FRAME BY FRAME
    # --------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # --------------------------------------------------
        # YOLO TRACKING
        # --------------------------------------------------

        results = model.track(
            frame,
            conf=confidence,
            classes=vehicle_classes,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        # Draw Line 1 and Line 2
        cv2.line(
            frame,
            (0, line_1),
            (width, line_1),
            (255, 0, 0),
            2
        )

        cv2.line(
            frame,
            (0, line_2),
            (width, line_2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "LINE 1",
            (10, line_1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "LINE 2",
            (10, line_2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # PROCESS DETECTIONS
        # --------------------------------------------------

        for result in results:

            if result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, class_id, conf in zip(
                boxes,
                track_ids,
                class_ids,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                vehicle_name = vehicle_names.get(
                    class_id,
                    "vehicle"
                )

                vehicle_types[track_id] = vehicle_name

                # --------------------------------------------------
                # DRAW BOUNDING BOX
                # --------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                label = (
                    f"{vehicle_name} "
                    f"ID:{track_id} "
                    f"{conf:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                # --------------------------------------------------
                # GET PREVIOUS POSITION
                # --------------------------------------------------

                previous_y = previous_positions.get(
                    track_id,
                    center_y
                )

                # --------------------------------------------------
                # LINE 1 CROSSING
                # --------------------------------------------------

                crossed_line_1 = (
                    previous_y < line_1
                    and center_y >= line_1
                )

                if (
                    crossed_line_1
                    and track_id not in line1_times
                ):

                    line1_times[track_id] = (
                        frame_number / fps
                    )

                # --------------------------------------------------
                # LINE 2 CROSSING
                # --------------------------------------------------

                crossed_line_2 = (
                    previous_y < line_2
                    and center_y >= line_2
                )

                if (
                    crossed_line_2
                    and track_id in line1_times
                    and track_id not in speed_results
                ):

                    line2_time = frame_number / fps

                    line1_time = line1_times[track_id]

                    time_taken = line2_time - line1_time

                    if time_taken > 0:

                        # Speed calculation
                        speed_mps = (
                            known_distance /
                            time_taken
                        )

                        speed_kmph = (
                            speed_mps * 3.6
                        )

                        speed_results[track_id] = (
                            speed_kmph
                        )

                        time_results[track_id] = (
                            time_taken
                        )

                # --------------------------------------------------
                # DISPLAY SPEED
                # --------------------------------------------------

                if track_id in speed_results:

                    speed = speed_results[track_id]

                    if speed > speed_limit:
                        status = "OVERSPEED"
                    else:
                        status = "NORMAL"

                    text = (
                        f"{speed:.1f} km/h - {status}"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                # Save current position
                previous_positions[track_id] = center_y

        # --------------------------------------------------
        # DISPLAY FRAME
        # --------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        # Progress
        progress = frame_number / total_frames

        progress_bar.progress(
            min(progress, 1.0)
        )

    # --------------------------------------------------
    # RELEASE VIDEO
    # --------------------------------------------------

    cap.release()

    progress_bar.empty()

    st.success("✅ Video processing completed!")


    # --------------------------------------------------
    # CREATE RESULTS TABLE
    # --------------------------------------------------

    if speed_results:

        rows = []

        for track_id, speed in speed_results.items():

            vehicle_type = vehicle_types.get(
                track_id,
                "vehicle"
            )

            time_taken = time_results.get(
                track_id,
                0
            )

            if speed > speed_limit:
                status = "OVERSPEED"
            else:
                status = "NORMAL"

            rows.append({
                "Vehicle ID": track_id,
                "Vehicle Type": vehicle_type,
                "Speed (km/h)": round(speed, 2),
                "Status": status,
                "Time Between Lines (sec)": round(
                    time_taken,
                    2
                ),
                "Distance Between Lines (m)": known_distance,
                "Speed Limit (km/h)": speed_limit
            })

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by Vehicle ID
        df = df.sort_values(
            by="Vehicle ID"
        ).reset_index(drop=True)


        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        st.subheader("📊 Vehicle Speed Results")

        st.dataframe(
            df,
            use_container_width=True
        )


        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        average_speed = df["Speed (km/h)"].mean()

        overspeed_count = (
            df["Status"] == "OVERSPEED"
        ).sum()

        normal_count = (
            df["Status"] == "NORMAL"
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Vehicles Measured",
            len(df)
        )

        col2.metric(
            "Average Speed",
            f"{average_speed:.2f} km/h"
        )

        col3.metric(
            "Overspeeding",
            overspeed_count
        )

        col4.metric(
            "Normal Speed",
            normal_count
        )


        # --------------------------------------------------
        # CSV EXPORT
        # --------------------------------------------------

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Speed Results as CSV",
            data=csv_data,
            file_name="vehicle_speed_results.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No vehicles crossed both lines. "
            "Try changing the line positions or confidence."
        )