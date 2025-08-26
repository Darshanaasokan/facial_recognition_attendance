import cv2
import face_recognition
import pickle
import os

# File to save encodings
ENCODINGS_FILE = "encodings.pkl"

# Load existing encodings if file exists
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

# Ask for student name
student_name = input("Enter student name: ")

# Open camera
video_capture = cv2.VideoCapture(0)

print("Press 'c' to capture face or 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Camera not working.")
        break

    # Show camera feed
    cv2.imshow("Register Face", frame)

    # Convert frame from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):  # Capture when 'c' is pressed
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]

                # Save encoding
                known_faces[student_name] = face_encoding
                with open(ENCODINGS_FILE, "wb") as f:
                    pickle.dump(known_faces, f)

                print(f"✅ Face of {student_name} registered successfully!")
                break
            else:
                print("⚠ Face detected but encoding failed. Try again.")
        else:
            print("⚠ No face detected. Please position your face clearly.")

    elif key == ord("q"):  # Quit without saving
        print("❌ Registration cancelled.")
        break

# Release camera and close windows
video_capture.release()
cv2.destroyAllWindows()
