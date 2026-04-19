import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model + labels
model = load_model("pose_model.h5")
le = pickle.load(open("label_encoder.pkl", "rb"))

# MediaPipe setup
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

# Function to extract keypoints
def extract_keypoints(frame, detector):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        kp = []
        for lm in result.hand_landmarks[0]:
            kp.extend([lm.x, lm.y, lm.z])
        return np.array(kp).reshape(1, -1)
    else:
        return None


# Start webcam
cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as detector:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        kp = extract_keypoints(frame, detector)

        if kp is not None:
            prediction = model.predict(kp)
            class_id = np.argmax(prediction)
            label = le.inverse_transform([class_id])[0]

            # Show prediction
            cv2.putText(frame, f"Sign: {label}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

        cv2.imshow("Sign Language Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()