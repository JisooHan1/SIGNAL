# realtime_predict.py

import cv2
import torch
import numpy as np
from collections import deque
from gesture_dataset.model import GestureBiLSTM
import mediapipe as mp

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureBiLSTM(input_dim=63, hidden_dim=128, output_dim=14).to(device)
model.load_state_dict(torch.load("gesture_bilstm.pt", map_location=device))
model.eval()

# MediaPipe 준비
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 실시간 입력 버퍼 (최근 10프레임)
seq_length = 10
buffer = deque(maxlen=seq_length)

# 라벨 매핑 (임시 예시: 원하는 출력으로 수정 가능)
label_map = {
    0: "D0X (None)",
    1: "B0A (One finger)",
    2: "B0B (Two fingers)",
    3: "G01 (Click)",
    4: "G02",
    5: "G03",
    6: "G04",
    7: "G05",
    8: "G06",
    9: "G07",
    10: "G08",
    11: "G09",
    12: "G10 (Zoom in)",
    13: "G11 (Zoom out)"
}

# 웹캠 시작
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
        print("✅ Buffer length:", len(buffer))
        print("🔍 First landmark x,y,z:", coords[0])  # 첫 번째 관절 위치
        print("📊 Mean Z of hand:", np.mean([p.z for p in lm.landmark]))


        mp_drawing.draw_landmarks(image, lm, mp_hands.HAND_CONNECTIONS)

        # 예측
        if len(buffer) == seq_length:
            input_tensor = torch.tensor([buffer], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                gesture = label_map.get(pred, "Unknown")

                # 🔥 여기서 제어 출력 (실제 제어는 send_command 등으로 대체)
                if gesture.startswith("G01"):
                    print("🟢 불 켜기 (Click)")
                elif gesture.startswith("D0X"):
                    print("⚪ 아무 동작 아님")
                else:
                    print(f"➡️ 인식된 제스처: {gesture}")

                # 화면에 출력
                cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SIGNAL - 실시간 제스처 인식", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
