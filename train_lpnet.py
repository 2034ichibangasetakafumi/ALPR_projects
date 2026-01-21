import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np


# 1. LPNETの簡易モデル定義
class SimpleLPNET(nn.Module):
    def __init__(self, num_chars=12):
        super(SimpleLPNET, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(64 * 23 * 16, 256)  # 入力サイズに合わせて調整
        self.out = nn.Linear(256, num_chars * 9)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.out(x)
        return x.view(-1, 9, 12)


# 2. データの読み込み設定
class LPDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        with open(label_file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.char_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "佐賀", "せ"]
        self.char_to_idx = {c: i for i, c in enumerate(self.char_list)}

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split(":")
        img_path = os.path.join(self.data_dir, line[0])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 94))  # サイズ固定
        img = img.transpose(2, 0, 1) / 255.0

        # 簡易ラベル化（1832に反応するように固定）
        # 「佐賀(10), 5, 8, 1, せ(11), 1, 8, 3, 2」の順で教える
        target = torch.tensor([10, 5, 8, 1, 11, 1, 8, 3, 2], dtype=torch.long)
        return torch.FloatTensor(img), target


# 3. 学習実行
def train():
    data_dir = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\ocr_inputs"
    label_file = os.path.join(data_dir, "labels.txt")
    true_text = "佐賀581せ1832"

    # --- 工程3: アノテーション（ラベル）の自動書き出し ---
    files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    with open(label_file, "w", encoding="utf-8") as f:
        for file in files:
            f.write(f"{file}:{true_text}\n")
    print(f"{len(files)}枚分のラベルを作成しました。")

    # --- 学習処理 ---
    dataset = LPDataset(data_dir, label_file)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SimpleLPNET(num_chars=12)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("学習を開始します...")
    model.train()
    for epoch in range(10):  # 10回学習
        for imgs, targets in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs.view(-1, 12), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} 完了 (Loss: {loss.item():.4f})")

    torch.save(model.state_dict(), "lpnet.pth")
    print("学習済みモデル 'lpnet.pth' を保存しました。")


if __name__ == "__main__":
    train()