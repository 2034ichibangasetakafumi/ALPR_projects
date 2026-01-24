import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO


# ==========================================
# 1. MODEL DEFINITION (認識モデルの定義)
# ==========================================
class SimpleLPNET(torch.nn.Module):
    """
    LPRNET: 切り出されたプレート画像から文字を読み取るためのモデル。
    CNNを使用して画像の特徴を抽出し、文字の並びを予測します。
    """

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


# ==========================================
# 2. CONFIGURATION (設定)
# ==========================================
FRAME_DIR = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\frames"
MODEL_PT = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\best.pt"
LPNET_PTH = "lpnet.pth"
OCR_OUT = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\ocr_inputs"
CHAR_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "佐賀", "せ"]

# 出力先フォルダの作成
os.makedirs(OCR_OUT, exist_ok=True)


# ==========================================
# 3. HELPER FUNCTIONS (補助関数)
# ==========================================

def preprocess_for_lpnet(plate_img):
    """
    OpenCVの画像をLPRNET（PyTorch）に入力できる形式に変換する。

    【なぜ関数化するのか？】
    - リサイズ、正規化、次元入れ替えという「お決まりの処理」を独立させることで、
      メインループの可読性を高めるためです。
    """
    # 認識精度を安定させるためにサイズを固定 (Width=64, Height=94)
    img = cv2.resize(plate_img, (64, 94))
    # [H, W, C] -> [C, H, W] へ変換し、0.0~1.0に正規化
    img_tensor = torch.FloatTensor(img.transpose(2, 0, 1) / 255.0).unsqueeze(0)
    return img_tensor


def recognize_plate_text(lpnet, img_tensor, char_list):
    """
    LPRNETモデルを使用して、画像から文字列を予測する。
    """
    with torch.no_grad():
        outputs = lpnet(img_tensor)
        # 各文字位置（9箇所）で最も確率が高い文字の番号を取得
        predict_indices = torch.argmax(outputs, dim=2)[0]

    # 番号を実際の文字に変換して結合
    return "".join([char_list[idx] for idx in predict_indices])


# ==========================================
# 4. MAIN PIPELINE (メイン実行処理)
# ==========================================
def main():
    # --- モデルのロード ---
    # YOLO: 画像内の「どこに」プレートがあるかを検出する (Object Detection)
    yolo_model = YOLO(MODEL_PT)

    # LPRNET: 見つかったプレートの「中身」が何かを認識する (OCR)
    lpnet_model = SimpleLPNET(num_chars=len(CHAR_LIST))
    lpnet_model.load_state_dict(torch.load(LPNET_PTH))
    lpnet_model.eval()

    # --- 画像の順次処理 ---
    files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith((".jpg", ".png"))])

    for frame_idx, file in enumerate(files):
        # 【Input】 フレームの読み込み
        frame_path = os.path.join(FRAME_DIR, file)
        frame = cv2.imread(frame_path)
        if frame is None: continue

        # 【Step 1: Detection】 YOLOでプレートを探す
        # conf=0.25 は確信度25%以上のものだけを拾う設定
        results = yolo_model(frame, conf=0.25)

        for i, res in enumerate(results[0].boxes):
            # 検出された座標 (x1, y1, x2, y2) を取得
            x1, y1, x2, y2 = map(int, res.xyxy[0])

            # 【Step 2: Cropping】 プレート部分を切り抜く
            # 境界ギリギリだと認識しにくいため、上下左右に2ピクセルの余白(Margin)を持たせる
            plate_img = frame[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2]
            if plate_img.size == 0: continue

            # 【Step 3: Recognition】 切り抜いた画像をLPRNETで文字にする
            # 前処理 (OpenCV -> Tensor)
            img_tensor = preprocess_for_lpnet(plate_img)
            # 推論 (Tensor -> Text)
            text = recognize_plate_text(lpnet_model, img_tensor, CHAR_LIST)

            # 【Output】 結果の表示と保存
            print(f"Frame: {file} | Plate {i}: {text}")

            # 後の学習用やデバッグ用に、切り抜いたプレート画像を保存
            save_name = f"f{frame_idx}_p{i}.png"
            cv2.imwrite(os.path.join(OCR_OUT, save_name), plate_img)


if __name__ == "__main__":
    main()