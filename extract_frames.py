import cv2
import os

VIDEO_PATH = r"C:\Users\user\Desktop\1768150033193_0.mp4"
OUT_DIR = "frames"
START_FRAME = 0    # 開始したいフレーム番号
END_FRAME = 150     # 終了したいフレーム番号（例）
SKIP = 1           # 密度を上げるため1に設定

os.makedirs(OUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

# 開始位置までスキップ
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
frame_id = START_FRAME
save_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_id > END_FRAME:
        break

    if (frame_id - START_FRAME) % SKIP == 0:
        cv2.imwrite(f"{OUT_DIR}/frame_{save_id:05d}.jpg", frame)
        save_id += 1

    frame_id += 1

cap.release()
print("extract done")
