import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load data
X = np.load("X_keypoints.npy")
y = np.load("y_labels.npy")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

import pickle
pickle.dump(le, open("label_encoder.pkl", "wb"))

#data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Build model
model_pose = Sequential([
    Input(shape=(63,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model_pose.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model_pose.fit(X_train, y_train, epochs=20, batch_size=32)

#evaluation
loss_pose, acc_pose = model_pose.evaluate(X_test, y_test)

print("Pose Model Accuracy:", acc_pose)

model_pose.save("pose_model.h5")