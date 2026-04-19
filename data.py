from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

X = np.load("X_keypoints.npy")
y = np.load("y_labels.npy")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)