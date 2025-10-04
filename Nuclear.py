import cv2
import face_recognition
import os
import numpy as np

# --- 1. Initialization and setup ---

# Folder containing authorized face images
AUTHORIZED_FACES_DIR = "authorized_faces"

# Lists to store known (authorized) face encodings and their names
known_face_encodings = []
known_face_names = []

# --- 2. Load and encode authorized faces ---

def load_authorized_faces():
    """
    Loads face images from the specified folder, encodes facial features,
    and stores them in the global lists.
    """
    print(f"Loading authorized faces from '{AUTHORIZED_FACES_DIR}'...")

    if not os.path.exists(AUTHORIZED_FACES_DIR):
        print(f"Error: Folder '{AUTHORIZED_FACES_DIR}' does not exist.")
        print("Please create it and put authorized person images inside.")
        return

    for filename in os.listdir(AUTHORIZED_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(AUTHORIZED_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)

            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)

                # Use the file name (without extension) as the person's name
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"  - Loaded successfully: {name}")
            else:
                print(f"  - Warning: No face found in {filename}, skipped.")

# Execute loading function
load_authorized_faces()
if not known_face_names:
    print("\nWarning: No authorized faces found. All faces will be marked as 'Unauthorized'.")
print(f"Database loaded. {len(known_face_names)} authorized person(s) found.")


# --- 3. Main loop: Real-time video processing ---

# Open the camera (0 = default webcam, 1 = external camera, etc.)
cap = cv2.VideoCapture(1)
print("Camera started. Press 'q' to quit.")

# Performance optimization: process every other frame
process_this_frame = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read from camera.")
        break

    # --- Optimization 1: Downscale frame for faster recognition ---
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # --- Optimization 2: Skip alternate frames ---
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unauthorized"

            # Find the best match among known faces
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # --- Display results ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale face coordinates back to the original frame size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        color = (0, 255, 0) if name != "Unauthorized" else (0, 0, 255)

        # Draw face rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw label with background rectangle
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Authorization Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("Program closed.")
