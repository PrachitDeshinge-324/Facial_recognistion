import cv2
from retinaface import RetinaFace
from deepface import DeepFace
import numpy as np

# --- 1. Load Face Recognition Model and Database ---
# It's crucial to load the model only once
model = DeepFace.build_model('ArcFace')

# In a real application, you would load your database embeddings here
# For this example, we'll use a simple dictionary to store embeddings
face_database = {
    "person_A": DeepFace.represent(img_path="./face_database/A/image1.jpg", model_name='ArcFace', detector_backend='retinaface')[0]["embedding"],
    "person_A": DeepFace.represent(img_path="./face_database/A/image2.jpg", model_name='ArcFace', detector_backend='retinaface')[0]["embedding"],
    "person_A": DeepFace.represent(img_path="./face_database/A/image3.jpg", model_name='ArcFace', detector_backend='retinaface')[0]["embedding"]
}

# --- 2. Real-time Video Processing ---
cap = cv2.VideoCapture("../Facial Recognision/video/03_09_2025_face_recognition.mp4") # or your CCTV stream URL

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # --- 3. Face Detection ---
        # We use RetinaFace for its high accuracy
        detected_faces = RetinaFace.detect_faces(frame)

        for key in detected_faces.keys():
            identity = detected_faces[key]
            facial_area = identity["facial_area"]
            x1, y1, x2, y2 = facial_area

            # --- 4. Face Recognition ---
            # Extract the face from the frame
            face_img = frame[y1:y2, x1:x2]

            try:
                # Get the embedding for the detected face
                embedding = DeepFace.represent(face_img, model_name='ArcFace', enforce_detection=False, detector_backend='skip')[0]["embedding"]

                # --- 5. Database Matching ---
                best_match_name = "Unknown"
                best_match_score = 0.5 # Threshold for ArcFace with cosine distance

                for name, db_embedding in face_database.items():
                    distance = np.linalg.norm(np.array(embedding) - np.array(db_embedding))
                    # Note: DeepFace's verify function does this internally. For real-time,
                    # manual comparison is often more flexible.
                    # This is a simplified distance calculation. For ArcFace, cosine similarity is better.

                    # Using DeepFace.verify for a more accurate comparison:
                    result = DeepFace.verify(face_img, db_embedding, model_name='ArcFace', enforce_detection=False, detector_backend='skip')

                    if result["verified"] and result["distance"] < best_match_score:
                        best_match_score = result["distance"]
                        best_match_name = name

                # --- 6. Visualization ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, best_match_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                # This will happen if a face is detected but not recognized (e.g., too blurry)
                pass

    except Exception as e:
        # This will happen if no faces are detected in the frame
        pass

    cv2.imshow("Real-time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()