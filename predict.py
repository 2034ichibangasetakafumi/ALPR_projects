import torch
import cv2
import numpy as np


# 1. モデルの定義（学習時と同じ構造）
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


# 2. 推論の実行
def predict(image_path):
    char_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "佐賀", "せ"]

    # モデルの準備
    model = SimpleLPNET(num_chars=12)
    model.load_state_dict(torch.load("lpnet.pth"))
    model.eval()

    # 画像の読み込みと加工
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 94))
    img_tensor = torch.FloatTensor(img.transpose(2, 0, 1) / 255.0).unsqueeze(0)

    # 予測
    with torch.no_grad():
        outputs = model(img_tensor)
        # 各文字位置で最も確率が高い数字を選択
        predict_indices = torch.argmax(outputs, dim=2)[0]

    # 数字を文字に変換
    result = "".join([char_list[i] for i in predict_indices])
    print(f"画像 {image_path} の認識結果: {result}")


if __name__ == "__main__":
    # テストしたい画像パスを指定
    predict(r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\ocr_inputs\f63_p0.png")