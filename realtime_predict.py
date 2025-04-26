import cv2
import torch
import numpy as np
import time
from collections import deque
from gesture_dataset.model import GestureCNNBiLSTM
import mediapipe as mp

# 설정
CONFIDENCE_THRESHOLD = 0.6  # softmax 확신도 기준
STABLE_THRESHOLD = 3        # 같은 예측 몇 번 연속이면 확정
HOLD_TIME = 2.0             # 확정 후 유지 시간 (초)
COOLDOWN_TIME = 1.5         # 다음 제스처 예측까지 대기 시간

# 모델
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureCNNBiLSTM(input_dim=63, cnn_out=128, lstm_hidden=128, output_dim=15).to(device)
model.load_state_dict(torch.load("gesture_cnn_bilstm.pt", map_location=device))
model.eval()

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 버퍼 & 라벨
seq_length = 10
buffer = deque(maxlen=seq_length)

label_map = {
    0: "Unknown",
    1: "D0X (None)",
    2: "B0A (pointing with one finger)",
    3: "B0B (pointing with two fingers)",
    4: "G01 (Click with one finger)",
    5: "G02 (Click with two fingers)",
    6: "G03 (Throw up)",
    7: "G04 (Throw down)",
    8: "G05 (Throw left)",
    9: "G06 (Throw right)",
    10: "G07 (Open twice)",
    11: "G08 (Double click with one finger)",
    12: "G09 (Double click with two fingers)",
    13: "G10 (Zoom in)",
    14: "G11 (Zoom out)"
}

# 상태 변수
last_pred, stable_pred = None, None
pred_count = 0
last_confirm_time = time.time()
last_output_time = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        coords = [(p.x, p.y, p.z) for p in lm.landmark]
        buffer.append(np.array(coords).flatten())
        mp_drawing.draw_landmarks(image, lm, mp_hands.HAND_CONNECTIONS)

        # 예측 처리
        if len(buffer) == seq_length:
            input_tensor = torch.tensor([buffer], dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                pred = pred.item()
                conf = conf.item()

                now = time.time()

                # 확신도 낮으면 무시
                if conf < CONFIDENCE_THRESHOLD:
                    pred = -1  # invalid prediction

                # 예측 안정화 처리
                if pred == last_pred:
                    pred_count += 1
                else:
                    pred_count = 1
                    last_pred = pred

                # 예측이 안정됐고 cooldown 지났고 D0X가 아닐 때만 확정
                if (
                    pred_count >= STABLE_THRESHOLD and
                    pred != 1 and pred != -1 and
                    (now - last_output_time) > COOLDOWN_TIME
                ):
                    stable_pred = pred
                    last_confirm_time = now
                    last_output_time = now

                    confirmed_gesture = label_map.get(pred, "Unknown")
                    print(f"✅ 인식 확정: {confirmed_gesture}")

                    # 행동 예시
                    if confirmed_gesture.startswith("G01"):
                        print("🟢 불 켜기")
                    else:
                        print(f"➡️ 제스처: {confirmed_gesture}")

    # 화면 출력 처리
    if stable_pred is not None and (time.time() - last_confirm_time < HOLD_TIME):
        display_text = label_map.get(stable_pred, "...")
    else:
        display_text = "..."

    cv2.putText(image, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SIGNAL - 실시간 제스처 인식 (강건 모드)", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
