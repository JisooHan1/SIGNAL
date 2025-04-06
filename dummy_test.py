import cv2
import mediapipe as mp
import numpy as np
import time

SEQUENCE_LENGTH = 30
sequence_buffer = []

# 모델 불러오기 예시 (여기선 가짜)
def load_model():
    return lambda x: "turn_on_light"  # 항상 같은 결과 내는 더미 모델

model = load_model()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence_buffer.append(landmarks)
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_seq = np.array(sequence_buffer)[np.newaxis, ...]
                prediction = model(input_seq)
                print("🖐 제스처 인식 결과:", prediction)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
