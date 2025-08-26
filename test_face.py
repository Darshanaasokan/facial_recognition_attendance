import cv2
import face_recognition

# Open the webcam
video_capture = cv2.VideoCapture(0)

print("Press 'q' to quit the test window.")

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Convert the frame to RGB (face_recognition works on RGB, OpenCV uses BGR)
    rgb_frame = frame[:, :, ::-1]

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw rectangles around faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Face Test', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
