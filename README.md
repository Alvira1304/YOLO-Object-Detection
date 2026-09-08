# YOLO Object Detection, Tracking & Speed Estimation

## 📌 Project Overview

This project is a Computer Vision application that uses YOLO
for vehicle detection and ByteTrack for object tracking.

The system can detect vehicles, track them, count vehicles
crossing a virtual line, estimate their speed, detect
overspeeding vehicles, and export results as a CSV file.

## 🚗 Features

- Vehicle detection using YOLO
- Vehicle tracking using ByteTrack
- Car, motorcycle, bus, and truck detection
- Vehicle counting using a virtual line
- Vehicle speed estimation
- Overspeed detection
- Traffic statistics
- CSV export of speed results
- Streamlit web interface

## 🧠 Technologies Used

- Python
- YOLO
- Ultralytics
- ByteTrack
- OpenCV
- Streamlit
- Pandas
- NumPy

## 🔄 Project Workflow

Input Image / Video
        ↓
YOLO Object Detection
        ↓
Vehicle Detection
        ↓
ByteTrack Object Tracking
        ↓
Virtual Line Crossing
        ↓
Vehicle Counting
        ↓
Two-Line Speed Estimation
        ↓
Overspeed Detection
        ↓
Traffic Statistics
        ↓
CSV Report

## 🎯 Vehicle Classes

The system focuses on four vehicle classes:

- Car
- Motorcycle
- Bus
- Truck

## 📊 Vehicle Counting

A virtual line is placed on the road.

When a tracked vehicle crosses the line, the system
counts that vehicle.

Tracking prevents the same vehicle from being counted
multiple times across consecutive video frames.

## 🚀 Speed Estimation

Two virtual lines are placed in the video.

When a tracked vehicle crosses Line 1, its time is recorded.

When the same vehicle crosses Line 2, the second time is recorded.

Speed is calculated using:

Speed = Distance / Time

The result is converted from metres per second
to kilometres per hour.

## 🚦 Overspeed Detection

The user can set a speed limit.

If the estimated vehicle speed is greater than the
selected speed limit, the vehicle is classified as:

OVERSPEED

Otherwise:

NORMAL

## 📁 CSV Export

The system can export vehicle speed results as a CSV file.

The CSV contains:

- Vehicle ID
- Vehicle Type
- Speed
- Status
- Time between lines
- Distance between lines
- Speed limit

## ⚠️ Speed Calibration

Speed estimation depends on the real-world distance
between the two virtual lines.

For accurate real-world speed measurement, the distance
between the lines should be calibrated using a known
physical distance.

## 💻 How to Run

### Install dependencies

```bash
pip install -r requirements.txt