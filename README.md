# Facial Recognition Attendance System

This project is a **prototype classroom attendance system** built with Python, OpenCV, and the `face_recognition` library.  
It uses facial recognition to automatically mark students present or absent during a class session.

---

## 🚀 Features
- Registers student faces using webcam (one-time registration).
- Simulates a **45-second class session** (instead of 45 minutes for demo).
- Takes snapshots every **15 seconds** (3 times total).
- Marks student **Present** if detected at least once, otherwise **Absent**.
- Saves results to a **CSV file** for easy tracking.

---

## 🛠️ Tech Stack
- Python
- OpenCV
- face_recognition
- Pandas
- Numpy
