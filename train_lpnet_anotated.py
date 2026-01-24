import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np


# ==========================================
# 1. MODEL DEFINITION (モデルの構造)
# ==========================================
class SimpleLPNET(nn.Module):
    """
    LPRNET（ナンバープレート認識）の構造を模した簡易CNNモデル。
    【役割】画像から特徴を抽出し、どの文字が書かれているかを分類します。
    """

    def __init__(self, num_chars=12):
        super(SimpleLPNET, self).__init__()
        # 特徴抽出層: 畳み込み(Conv)とプーリング(Maxpool)で画像の特徴を絞り込む
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 結合層: 抽出した特徴を1次元に並べて、最終的な文字の判定に繋げる
        # 入力画像サイズ (94x64) から MaxPoolを2回経て (23x16) に縮小された状態
        self.fc = nn.Linear(64 * 23 * 16, 256)
        # 出力層: 9文字分 × 各文字の確率(12種類) を出力
        self.out = nn.Linear(256, num_chars * 9)

    def forward(self, x):
        """データの流れを定義"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 2次元画像を1次元ベクトルに平坦化(Flatten)
        x = self.fc(x)
        x = self.out(x)
        # 出力形状を [バッチサイズ, 9文字, 12種類の確率] に整形
        return x.view(-1, 9, 12)


# ==========================================
# 2. DATASET MANAGEMENT (データの読み込みと加工)
# ==========================================
class LPDataset(Dataset):
    """
    PyTorchがデータを1枚ずつ取り出せるようにするための管理クラス。

    【なぜDatasetクラスを作るのか？】
    - 大量の画像を一度にメモリに乗せるとパンクするため、「必要な時に1枚ずつ」
      読み込み・加工(リサイズや正規化)を行う仕組みが必要です。
    """

    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        with open(label_file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        # 文字と数字（インデックス）の対応表
        self.char_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "佐賀", "せ"]
        self.char_to_idx = {c: i for i, c in enumerate(self.char_list)}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        """
        【Input】 画像パスとラベルの読み込み
        【Processing】 画像のリサイズ、色の並び替え(HWC->CHW)、正規化(0-1)
        【Output】 AIが計算できる形式のテンソルデータ
        """
        line = self.lines[idx].strip().split(":")
        img_path = os.path.join(self.data_dir, line[0])

        # 画像の読み込みと前処理 (OpenCVを使用)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 94))  # モデルが期待するサイズに固定
        img = img.transpose(2, 0, 1) / 255.0  # PyTorch用の次元順序に変更

        # 教師データ(Target): 正解の文字インデックスを並べる
        # 「佐賀(10), 5, 8, 1, せ(11), 1, 8, 3, 2」という正解ラベルを生成
        target = torch.tensor([10, 5, 8, 1, 11, 1, 8, 3, 2], dtype=torch.long)

        return torch.FloatTensor(img), target


# ==========================================
# 3. TRAINING LOGIC (学習の実行)
# ==========================================
def prepare_labels(data_dir, label_file, true_text):
    """
    学習の前に、画像と正解テキストを紐付けるラベルファイルを作成します。
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    with open(label_file, "w", encoding="utf-8") as f:
        for file in files:
            f.write(f"{file}:{true_text}\n")
    print(f"事前準備完了: {len(files)}枚分のラベルを作成しました。")


def train_model():
    """
    学習のメインループ。
    【なぜ関数化するのか？】
    - 「アノテーション作成」「データの準備」「学習ループ」「モデル保存」という
      一連のワークフローを1つの手続きとして整理し、他から呼び出しやすくするためです。
    """
    # 設定
    data_dir = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\ocr_inputs"
    label_file = os.path.join(data_dir, "labels.txt")
    true_text = "佐賀581せ1832"

    # --- Step 1: ラベルファイルの作成 ---
    prepare_labels(data_dir, label_file, true_text)

    # --- Step 2: データローダーの準備 ---
    # DataLoaderは、データを「バッチサイズ」ごとにまとめてモデルに投入する役目。
    dataset = LPDataset(data_dir, label_file)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # --- Step 3: モデル、最適化アルゴリズム、損失関数の設定 ---
    model = SimpleLPNET(num_chars=12)
    # Adam: 学習率を自動調整してくれる賢い最適化手法
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # CrossEntropyLoss: 「正解とどれだけズレているか」を計算する（多クラス分類用）
    criterion = nn.CrossEntropyLoss()

    # --- Step 4: 学習ループ ---
    print("学習を開始します...")
    model.train()  # モデルを学習モードに設定

    for epoch in range(10):  # 同じデータを10回繰り返し学習
        total_loss = 0
        for imgs, targets in loader:
            # 1. 勾配の初期化（前回の計算結果をクリア）
            optimizer.zero_grad()

            # 2. 推論 (Forward)
            outputs = model(imgs)

            # 3. 誤差計算 (Loss)
            # view(-1, 12) は損失計算関数が受け取れる形状(2次元)に平坦化しています
            loss = criterion(outputs.view(-1, 12), targets.view(-1))

            # 4. 誤差逆伝播 (Backward) : どのパラメータを直すべきか計算
            loss.backward()

            # 5. パラメータ更新 (Step) : 実際にモデルの重みを修正
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/10 - Average Loss: {total_loss / len(loader):.4f}")

    # --- Step 5: 保存 (Output) ---
    torch.save(model.state_dict(), "lpnet.pth")
    print("学習完了: 'lpnet.pth' として保存しました。")


# ==========================================
# ENTRY POINT (実行)
# ==========================================
if __name__ == "__main__":
    train_model()