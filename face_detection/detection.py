from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

img_path = "./face_database/A/image1.jpg"
img = cv2.imread(img_path)
obj = RetinaFace.detect_faces(img_path)
for key in obj.keys():
    identity = obj[key]
    facial_area = identity["facial_area"]
    x1, y1, x2, y2 = facial_area
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(10,10))
plt.imshow(img[:,:,::-1])
plt.axis('off')
plt.show()

from deepface import DeepFace
obj = DeepFace.verify(img1_path = "./face_database/A/image1.jpg", img2_path = "./face_database/A/image2.jpg", model_name = 'ArcFace', detector_backend = 'retinaface')
print(obj)