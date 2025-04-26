import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_hand_landmarks_sequence(base_dir, video_id, start, end):
    sequence = []
    for i in range(start, end + 1):
        filename = f"{video_id}_{i:06d}.jpg"
        img_path = os.path.join(base_dir, video_id, filename)

        if not os.path.exists(img_path):
            continue  # 프레임 파일이 없으면 스킵

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            coords = [(p.x, p.y, p.z) for p in lm.landmark]
            sequence.append(np.array(coords).flatten())
        else:
            sequence.append(np.zeros(21 * 3))  # 손 인식 실패 시 0벡터로 채움

    return np.array(sequence)
