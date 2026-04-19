import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = "asl_alphabet_train"

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

X = []
y = []

def extract_keypoints(image_path, detector):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        kp = []
        for lm in result.hand_landmarks[0]:
            kp.extend([lm.x, lm.y, lm.z])
        return kp
    else:
        return [0]*63


with vision.HandLandmarker.create_from_options(options) as detector:
    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            kp = extract_keypoints(path, detector)
            if kp is not None:
                X.append(kp)
                y.append(label)

X = np.array(X)
y = np.array(y)

np.save("X_keypoints.npy", X)
np.save("y_labels.npy", y)

print("Saved keypoints:", X.shape)