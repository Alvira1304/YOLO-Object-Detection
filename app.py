import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd

from ultralytics import YOLO
from collections import Counter


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Traffic Monitoring System",
    page_icon="🚗",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🚗 Real-Time Traffic Monitoring & Analytics")

st.write(
    "Detect, track, and count vehicles using YOLO and ByteTrack. "
    "Analyze traffic flow using vehicle statistics and traffic density."
)


# ==========================================================
# LOAD YOLO MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙️ Detection Settings")


confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)


# COCO vehicle classes
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]


mode = st.sidebar.radio(
    "Select Input",
    ["Image", "Video"]
)


# ==========================================================
# FUNCTION: TRAFFIC DENSITY
# ==========================================================

def get_traffic_density(vehicles_per_minute):

    if vehicles_per_minute < 10:
        return "🟢 Low Traffic"

    elif vehicles_per_minute < 25:
        return "🟡 Medium Traffic"

    else:
        return "🔴 High Traffic"


# ==========================================================
# IMAGE MODE
# ==========================================================

if mode == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # --------------------------------------------------
        # READ IMAGE
        # --------------------------------------------------

        file_bytes = uploaded_file.read()

        image = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )


        # --------------------------------------------------
        # ORIGINAL IMAGE
        # --------------------------------------------------

        st.subheader("📷 Original Image")

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )


        # --------------------------------------------------
        # YOLO DETECTION
        # --------------------------------------------------

        results = model(
            image,
            conf=confidence,
            classes=vehicle_classes,
            verbose=False
        )


        # --------------------------------------------------
        # DRAW DETECTIONS
        # --------------------------------------------------

        annotated_image = results[0].plot()


        st.subheader("🎯 Detection Result")

        st.image(
            cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )


        # --------------------------------------------------
        # COUNT VEHICLES
        # --------------------------------------------------

        detected_objects = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = result.names[class_id]

                detected_objects.append(class_name)


        vehicle_counts = Counter(detected_objects)


        # --------------------------------------------------
        # IMAGE STATISTICS
        # --------------------------------------------------

        st.subheader("📊 Vehicle Statistics")

        total_vehicles = sum(
            vehicle_counts.values()
        )


        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "🚗 Cars",
                vehicle_counts.get("car", 0)
            )

        with col2:
            st.metric(
                "🚌 Buses",
                vehicle_counts.get("bus", 0)
            )

        with col3:
            st.metric(
                "🚛 Trucks",
                vehicle_counts.get("truck", 0)
            )

        with col4:
            st.metric(
                "🏍️ Motorcycles",
                vehicle_counts.get("motorcycle", 0)
            )

        with col5:
            st.metric(
                "🚦 Total",
                total_vehicles
            )


# ==========================================================
# VIDEO MODE
# ==========================================================

else:

    uploaded_video = st.file_uploader(
        "Upload a traffic video",
        type=["mp4", "avi", "mov"]
    )


    if uploaded_video is not None:

        # --------------------------------------------------
        # COUNTING LINE
        # --------------------------------------------------

        st.sidebar.subheader("📏 Counting Line")


        line_y = st.sidebar.slider(
            "Line Position",
            min_value=50,
            max_value=450,
            value=300,
            step=5
        )


        st.sidebar.write(
            f"Line Position: **{line_y}**"
        )


        # --------------------------------------------------
        # START BUTTON
        # --------------------------------------------------

        start_button = st.button(
            "▶️ Start Traffic Analysis"
        )


        if start_button:

            # --------------------------------------------------
            # TEMPORARY VIDEO FILE
            # --------------------------------------------------

            temp_video = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )


            temp_video.write(
                uploaded_video.read()
            )

            temp_video.close()


            # --------------------------------------------------
            # OPEN VIDEO
            # --------------------------------------------------

            cap = cv2.VideoCapture(
                temp_video.name
            )


            if not cap.isOpened():

                st.error(
                    "Could not open video."
                )


            else:

                # --------------------------------------------------
                # VIDEO INFORMATION
                # --------------------------------------------------

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )


                # Video duration
                if fps > 0:
                    video_duration = (
                        total_frames / fps
                    )
                else:
                    video_duration = 0


                st.info(
                    f"Video: {width} × {height} | "
                    f"FPS: {fps:.2f} | "
                    f"Frames: {total_frames} | "
                    f"Duration: {video_duration:.2f} seconds"
                )


                # --------------------------------------------------
                # TRACKING VARIABLES
                # --------------------------------------------------

                previous_positions = {}

                counted_ids = set()

                vehicle_counts = Counter()

                current_counts = Counter()


                # --------------------------------------------------
                # DISPLAY
                # --------------------------------------------------

                frame_placeholder = st.empty()

                progress_bar = st.progress(0)


                frame_number = 0


                # ==================================================
                # PROCESS VIDEO
                # ==================================================

                while True:

                    success, frame = cap.read()


                    if not success:
                        break


                    frame_number += 1


                    # --------------------------------------------------
                    # YOLO + BYTETRACK
                    # --------------------------------------------------

                    results = model.track(
                        frame,
                        conf=confidence,
                        classes=vehicle_classes,
                        tracker="bytetrack.yaml",
                        persist=True,
                        verbose=False
                    )


                    result = results[0]


                    # --------------------------------------------------
                    # DRAW YOLO BOXES
                    # --------------------------------------------------

                    annotated_frame = result.plot()


                    # --------------------------------------------------
                    # DRAW COUNTING LINE
                    # --------------------------------------------------

                    cv2.line(
                        annotated_frame,
                        (0, line_y),
                        (width, line_y),
                        (0, 255, 255),
                        3
                    )


                    cv2.putText(
                        annotated_frame,
                        "COUNTING LINE",
                        (20, line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )


                    # --------------------------------------------------
                    # RESET CURRENT FRAME COUNT
                    # --------------------------------------------------

                    current_counts = Counter()


                    # ==================================================
                    # TRACK VEHICLES
                    # ==================================================

                    if (
                        result.boxes is not None
                        and result.boxes.id is not None
                    ):

                        boxes = result.boxes


                        # Tracking IDs

                        track_ids = (
                            boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )


                        # Class IDs

                        class_ids = (
                            boxes.cls
                            .int()
                            .cpu()
                            .tolist()
                        )


                        # Bounding boxes

                        coordinates = (
                            boxes.xyxy
                            .cpu()
                            .tolist()
                        )


                        # --------------------------------------------------
                        # PROCESS EACH VEHICLE
                        # --------------------------------------------------

                        for (
                            track_id,
                            class_id,
                            box
                        ) in zip(
                            track_ids,
                            class_ids,
                            coordinates
                        ):

                            x1, y1, x2, y2 = map(
                                int,
                                box
                            )


                            # Center point

                            center_x = int(
                                (x1 + x2) / 2
                            )

                            center_y = int(
                                (y1 + y2) / 2
                            )


                            # Vehicle name

                            class_name = (
                                result.names[
                                    class_id
                                ]
                            )


                            # --------------------------------------------------
                            # CURRENT FRAME COUNT
                            # --------------------------------------------------

                            current_counts[
                                class_name
                            ] += 1


                            # --------------------------------------------------
                            # DRAW CENTER POINT
                            # --------------------------------------------------

                            cv2.circle(
                                annotated_frame,
                                (
                                    center_x,
                                    center_y
                                ),
                                5,
                                (255, 0, 255),
                                -1
                            )


                            # --------------------------------------------------
                            # CHECK LINE CROSSING
                            # --------------------------------------------------

                            if track_id in previous_positions:

                                previous_y = (
                                    previous_positions[
                                        track_id
                                    ]
                                )


                                # Vehicle moving DOWN

                                crossed_line = (
                                    previous_y < line_y
                                    and center_y >= line_y
                                )


                                if (
                                    crossed_line
                                    and track_id
                                    not in counted_ids
                                ):

                                    # Remember this ID

                                    counted_ids.add(
                                        track_id
                                    )


                                    # IMPORTANT:
                                    # Count the actual vehicle type

                                    vehicle_counts[
                                        class_name
                                    ] += 1


                            # Save current position

                            previous_positions[
                                track_id
                            ] = center_y


                    # ==================================================
                    # TRAFFIC ANALYTICS
                    # ==================================================

                    total_crossed = sum(
                        vehicle_counts.values()
                    )


                    # Calculate elapsed video time

                    if fps > 0:

                        elapsed_seconds = (
                            frame_number / fps
                        )

                    else:

                        elapsed_seconds = 0


                    elapsed_minutes = (
                        elapsed_seconds / 60
                    )


                    # Vehicles per minute

                    if elapsed_minutes > 0:

                        vehicles_per_minute = (
                            total_crossed
                            / elapsed_minutes
                        )

                    else:

                        vehicles_per_minute = 0


                    # Traffic density

                    traffic_density = (
                        get_traffic_density(
                            vehicles_per_minute
                        )
                    )


                    # ==================================================
                    # DISPLAY ANALYTICS ON VIDEO
                    # ==================================================

                    cv2.putText(
                        annotated_frame,
                        f"Vehicles Crossed: {total_crossed}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        annotated_frame,
                        f"Cars: {vehicle_counts.get('car', 0)}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        annotated_frame,
                        f"Buses: {vehicle_counts.get('bus', 0)}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        annotated_frame,
                        f"Trucks: {vehicle_counts.get('truck', 0)}",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        annotated_frame,
                        f"Motorcycles: {vehicle_counts.get('motorcycle', 0)}",
                        (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )


                    # --------------------------------------------------
                    # DISPLAY VIDEO
                    # --------------------------------------------------

                    frame_rgb = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    )


                    frame_placeholder.image(
                        frame_rgb,
                        use_container_width=True
                    )


                    # --------------------------------------------------
                    # PROGRESS
                    # --------------------------------------------------

                    if total_frames > 0:

                        progress = (
                            frame_number
                            / total_frames
                        )


                        progress_bar.progress(
                            min(
                                progress,
                                1.0
                            )
                        )


                # ==================================================
                # FINISH
                # ==================================================

                cap.release()

                progress_bar.progress(1.0)


                st.success(
                    "✅ Traffic analysis completed!"
                )


                # ==================================================
                # FINAL ANALYTICS
                # ==================================================

                st.subheader(
                    "📊 Final Traffic Analytics"
                )


                # Final counts

                cars = vehicle_counts.get(
                    "car",
                    0
                )

                buses = vehicle_counts.get(
                    "bus",
                    0
                )

                trucks = vehicle_counts.get(
                    "truck",
                    0
                )

                motorcycles = vehicle_counts.get(
                    "motorcycle",
                    0
                )


                total_vehicles = (
                    cars
                    + buses
                    + trucks
                    + motorcycles
                )


                # Final vehicles per minute

                if video_duration > 0:

                    final_vpm = (
                        total_vehicles
                        / (video_duration / 60)
                    )

                else:

                    final_vpm = 0


                final_density = (
                    get_traffic_density(
                        final_vpm
                    )
                )


                # ==================================================
                # METRICS
                # ==================================================

                col1, col2, col3, col4, col5 = st.columns(5)


                with col1:

                    st.metric(
                        "🚗 Cars",
                        cars
                    )


                with col2:

                    st.metric(
                        "🚌 Buses",
                        buses
                    )


                with col3:

                    st.metric(
                        "🚛 Trucks",
                        trucks
                    )


                with col4:

                    st.metric(
                        "🏍️ Motorcycles",
                        motorcycles
                    )


                with col5:

                    st.metric(
                        "🚦 Total",
                        total_vehicles
                    )


                # ==================================================
                # TRAFFIC FLOW
                # ==================================================

                st.subheader(
                    "🚦 Traffic Flow"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Vehicles / Minute",
                        f"{final_vpm:.2f}"
                    )


                with col2:

                    st.metric(
                        "Traffic Density",
                        final_density
                    )


                # ==================================================
                # VEHICLE DISTRIBUTION
                # ==================================================

                st.subheader(
                    "📈 Vehicle Distribution"
                )


                chart_data = pd.DataFrame(
                    {
                        "Vehicle Type": [
                            "Car",
                            "Bus",
                            "Truck",
                            "Motorcycle"
                        ],
                        "Count": [
                            cars,
                            buses,
                            trucks,
                            motorcycles
                        ]
                    }
                )


                st.bar_chart(
                    chart_data.set_index(
                        "Vehicle Type"
                    )
                )


                # ==================================================
                # FINAL TABLE
                # ==================================================

                st.subheader(
                    "📋 Traffic Summary"
                )


                summary_data = pd.DataFrame(
                    {
                        "Vehicle Type": [
                            "Car",
                            "Bus",
                            "Truck",
                            "Motorcycle",
                            "TOTAL"
                        ],
                        "Vehicles Crossed": [
                            cars,
                            buses,
                            trucks,
                            motorcycles,
                            total_vehicles
                        ]
                    }
                )


                st.dataframe(
                    summary_data,
                    use_container_width=True,
                    hide_index=True
                )