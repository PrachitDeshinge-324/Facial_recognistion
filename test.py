import insightface
import cv2
import numpy as np

# Initialize the InsightFace app with a pre-trained model
# The 'antilopev2' model is a common choice for general face analysis
app = insightface.app.FaceAnalysis(name='antelopev2')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load an image
img = cv2.imread('face_database/A/image1.jpg')

# Perform face detection and analysis
faces = app.get(img)

# Process detected faces
for face in faces:
    # Access bounding box coordinates
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    # Access keypoints (landmarks)
    kps = face.kps.astype(int)

    # Access face embedding (for recognition)
    embedding = face.embedding

    # Access gender and age (if available in the model)
    gender = face.gender
    age = face.age

    # Draw bounding box and keypoints on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for kp in kps:
        cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)

    # Print information
    print(f"Detected Face: BBox={bbox}, Gender={gender}, Age={age}")

# Display the image with detections
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()