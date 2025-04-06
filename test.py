# predict_live.py
import cv2
import torch
import numpy as np
import mediapipe as mp
from train import BiLSTM
import json
import os

# Configuration
SEQUENCE_LENGTH = 30
MODEL_PATH = "model/saved/gesture_model.pt"
LABEL_PATH = "data/labels.json"

# Load labels
with open(LABEL_PATH, 'r') as f:
    idx_to_label = {v: k for k, v in json.load(f).items()}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM(input_size=63, hidden_size=128, num_classes=len(idx_to_label)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize sequence buffer
buffer = []

cap = cv2.VideoCapture(0)
print("\n Real-time gesture recognition started. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        buffer.append(landmarks)
        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        if len(buffer) > SEQUENCE_LENGTH:
            buffer.pop(0)

        if len(buffer) == SEQUENCE_LENGTH:
            sequence = torch.tensor([buffer], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model(sequence)  # (1, num_classes)
                label_idx = torch.argmax(pred, dim=1).item()  # Predicted class index
                label = idx_to_label[label_idx]   # Convert number to text (e.g., 0 → "turn_on_light")
                confidence = torch.softmax(pred, dim=1)[0][label_idx].item()  # Prediction probability

            cv2.putText(image, f"{label} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # e.g., "turn_on_light (0.95)"

    else:
        buffer.clear()
        cv2.putText(image, "Show your hand", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Live Gesture Prediction", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n Gesture recognition ended.")