import cv2
import face_recognition
import numpy as np
import pickle
import csv
import time
from datetime import datetime
import os

# Load known faces (dictionary format)
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_face_encodings = list(data.values())
known_face_names = list(data.keys())

# Track detections across 3 snapshots
detections = {name: [] for name in known_face_names}

# Session setup (demo: 45 seconds total, 3 intervals of 15 seconds)
session_duration = 45
interval = 15
start_time = time.time()

cap = cv2.VideoCapture(0)

print("Class started! Attendance will be taken at 15s, 30s, 45s...\n")

snap_count = 0

while (time.time() - start_time) < session_duration and snap_count < 3:
    ret, frame = cap.read()
    if not ret:
        continue

    elapsed_time = int(time.time() - start_time)
    remaining = interval - (elapsed_time % interval)

    # Show countdown on camera
    cv2.putText(frame, f"Next snapshot in: {remaining}s",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendance Camera", frame)

    # Snapshot exactly at 15s, 30s, 45s
    if elapsed_time > 0 and elapsed_time % interval == 0:
        snap_count += 1
        print(f"\n📸 Taking snapshot {snap_count} at {elapsed_time}s...")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                detected_names.append(name)
                detections[name].append("Detected")
                print(f"✅ {name}: Detected")

        # For students not detected in this snapshot
        for name in known_face_names:
            if name not in detected_names:
                detections[name].append("Not Detected")
                print(f"❌ {name}: Not Detected")

        # Prevent multiple detections within same second
        time.sleep(1)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Final attendance decision
attendance = {}
for name, status_list in detections.items():
    if "Detected" in status_list:
        attendance[name] = "Present"
    else:
        attendance[name] = "Absent"

# Save attendance to CSV (append mode)
date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_exists = os.path.isfile("attendance.csv")

with open("attendance.csv", mode="a", newline="") as file:
    writer = csv.writer(file)

    # Write header only if file is new
    if not file_exists:
        writer.writerow(["Name", "Snapshot1", "Snapshot2", "Snapshot3", "Final Status", "Date"])

    for name, status_list in detections.items():
        while len(status_list) < 3:  # Pad missing snaps
            status_list.append("Not Detected")
        writer.writerow([name, status_list[0], status_list[1], status_list[2], attendance[name], date_str])

print("\n✅ Final Attendance saved to attendance.csv")
