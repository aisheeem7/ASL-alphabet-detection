# ASL Alphabet Detection

## Overview

This project detects American Sign Language (ASL) alphabets using:

* Pose-based model (hand keypoints using MediaPipe)
* CNN-based image classification
* Real-time webcam prediction system

---

## Features

* Real-time sign detection using webcam
* Pose-based detection using hand landmarks
* CNN model for image-based classification
* Comparative analysis between models

---

## Models Used

* Pose Model (MediaPipe keypoints + Dense Neural Network)
* CNN (Convolutional Neural Network)

---

## Results

### Pose Model Performance

* Accuracy: 74.91%
* Loss: ~0.91

Observations:

* Performs well using only hand keypoints
* Efficient and lightweight
* More stable across different backgrounds

---

### CNN Model Performance

* Accuracy: 67.72%
* Loss: ~1.85

Observations:

* High training accuracy (~99%) but lower test accuracy
* Indicates overfitting
* Sensitive to background variations

---

## Model Comparison

| Model      | Accuracy | Key Observation       |
| ---------- | -------- | --------------------- |
| Pose Model | 74.91%   | Better generalization |
| CNN        | 67.72%   | Overfitting observed  |

---

## Key Insights

* The pose-based model outperformed the CNN in this setup
* CNN achieved very high training accuracy but failed to generalize
* Hand keypoints provide a compact and robust feature representation
* This demonstrates the importance of feature selection over raw image input

---

## Dataset

ASL Alphabet Dataset (Kaggle):
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

* Approximately 87,000 images
* 29 classes (A–Z and additional symbols)
* Approximately 3000 images per class

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run real-time detection

python realtime_detection.py

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe

---

## Future Improvements

* Improve CNN generalization using data augmentation and dropout
* Extend pose model using temporal models such as LSTM
* Add sentence formation from detected alphabets
* Integrate text-to-speech functionality

---

## Author

Aishee Mukherjee
