import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO


# --- LPRNET モデル定義 ---
class SimpleLPNET(torch.nn.Module):
    def __init__(self, num_chars=12):
        super(SimpleLPNET, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, 1, 1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc = torch.nn.Linear(64 * 23 * 16, 256)
        self.out = torch.nn.Linear(256, num_chars * 9)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.out(x)
        return x.view(-1, 9, 12)


# ========================
# 設定・モデル読み込み
# ========================
FRAME_DIR = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\frames"
MODEL_PT = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\best.pt"
LPNET_PTH = "lpnet.pth"
OCR_OUT = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\ocr_inputs"
CHAR_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "佐賀", "せ"]

os.makedirs(OCR_OUT, exist_ok=True)

# YOLOとLPRNETの準備
yolo_model = YOLO(MODEL_PT)
lpnet = SimpleLPNET(num_chars=12)
lpnet.load_state_dict(torch.load(LPNET_PTH))
lpnet.eval()

# ========================
# メインループ
# ========================
for frame_idx, file in enumerate(sorted(os.listdir(FRAME_DIR))):
    frame = cv2.imread(os.path.join(FRAME_DIR, file))
    if frame is None: continue

    results = yolo_model(frame, conf=0.25)

    for i, res in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, res.xyxy[0])
        plate_img = frame[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2]
        if plate_img.size == 0: continue

        # 画像の読み込みと加工
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 94))
        img_tensor = torch.FloatTensor(img.transpose(2, 0, 1) / 255.0).unsqueeze(0)

        # LPRNETで文字認識
        with torch.no_grad():
            outputs = lpnet(img_tensor)
            predict_indices = torch.argmax(outputs, dim=2)[0]

        text = "".join([CHAR_LIST[idx] for idx in predict_indices])
        print(f"Frame {frame_idx} | Plate {i}: {text}")

        # 保存
        cv2.imwrite(os.path.join(OCR_OUT, f"f{frame_idx}_p{i}.png"), plate_img)