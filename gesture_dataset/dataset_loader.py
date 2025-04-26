
# gesture_dataset/dataset_loader.py
import os
import numpy as np
from .landmark_extractor import extract_hand_landmarks_sequence

def create_dataset(annotation_file, frames_base_dir, max_sequences=None):
    sequences, labels = [], []
    total, used = 0, 0

    with open(annotation_file, 'r') as f:
        for idx, line in enumerate(f):
            total += 1
            if max_sequences and idx >= max_sequences:
                break
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            video_id = parts[0]
            label = int(parts[2])
            start, end = int(parts[3]), int(parts[4])

            # ✅ 여기 출력!
            print(f"🎬 Processing {idx+1}: {video_id} ({start}~{end}) | label={label}")

            seq = extract_hand_landmarks_sequence(frames_base_dir, video_id, start, end)
            hand_detected_frames = sum(np.any(frame) for frame in seq)

            if hand_detected_frames >= 1:
                sequences.append(seq)
                labels.append(label)
                used += 1
            else:
                print(f"❌ Skipped (no hand detected): {video_id} {start}-{end}")

    print(f"📊 총 시도한 인스턴스 수: {total}")
    print(f"✅ 실제 사용된 시퀀스 수: {used}")
    return sequences, labels


# 1CM1_4_R_#229  video, 0
# B0B  label, 1
# 3  id, 2
# 2392  t_start, 3
# 2603  t_end, 4
# 212  frames 5






