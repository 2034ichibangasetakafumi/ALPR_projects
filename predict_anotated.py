import torch
import cv2
import numpy as np

# ==========================================
# 1. CONFIGURATION (設定)
# ==========================================
# 認識対象の文字リスト（ラベル）
# モデルの出力（数字のインデックス）をこの文字に変換します
CHAR_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "佐賀", "せ"]

# 学習済みモデルのパス
MODEL_PATH = "lpnet.pth"

# 画像処理設定
# OpenCVのresizeは (Width, Height) ですが、
# モデル内の計算 (64*23*16) から逆算すると、Height=94, Width=64 である必要があります。
IMG_WIDTH = 64
IMG_HEIGHT = 94


# ==========================================
# 2. MODEL DEFINITION (モデル定義)
# ==========================================
class SimpleLPNET(torch.nn.Module):
    """
    CNN (畳み込みニューラルネットワーク) モデルの定義クラス。
    学習時と同じ構造である必要があります。
    """

    def __init__(self, num_chars=12):
        super(SimpleLPNET, self).__init__()

        # 特徴抽出部 (Feature Extractor)
        # 画像から「線」「角」「丸み」などの特徴を取り出します。
        self.conv = torch.nn.Sequential(
            # [Step 1] 入力(3ch) -> 32種類のフィルターで特徴抽出 -> 画像サイズを1/2に圧縮
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),  # 活性化関数（不要な情報を捨てる）
            torch.nn.MaxPool2d(2, 2),  # プーリング（画像を縮小し、特徴を凝縮）

            # [Step 2] 32ch -> 64種類のフィルター -> さらに画像サイズを1/2に圧縮
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        # 分類部 (Classifier)
        # 抽出された特徴を使って、「どの文字か」を判断します。
        # 64(ch) * 23(H) * 16(W) は、2回のMaxPool(1/2 * 1/2 = 1/4)後のサイズです。
        # (Height: 94/4 ≒ 23, Width: 64/4 = 16)
        self.fc = torch.nn.Linear(64 * 23 * 16, 256)

        # 出力層: 最大9文字 × 文字種(12クラス) の確率を出力
        self.out = torch.nn.Linear(256, num_chars * 9)

    def forward(self, x):
        """
        データの流れ（順伝播）を定義します。
        """
        # 1. 画像から特徴マップを作成 [Batch, 64, 23, 16]
        x = self.conv(x)

        # 2. 一列に並べる (Flatten)
        # 2次元の画像データを、全結合層に入れるために1次元のベクトルに変換します。
        # x.size(0) はバッチサイズ（通常は1）を維持します。
        x = x.view(x.size(0), -1)

        # 3. 全結合層を通す
        x = self.fc(x)

        # 4. 最終出力
        x = self.out(x)

        # 5. 出力の整形 [Batch, 文字数(9), 文字クラス数(12)]
        # これにより、9つの位置それぞれでどの文字であるかの確率が得られます。
        return x.view(-1, 9, 12)


# ==========================================
# 3. PRE-PROCESSING (前処理)
# ==========================================
def preprocess_image(image_path):
    """
    画像を読み込み、PyTorchモデルに入力できる形式（テンソル）に変換する関数。

    【なぜ関数化するのか？】
    - OpenCV形式(H,W,C) から PyTorch形式(C,H,W) への変換は複雑で間違いやすいため、
      ロジックを分離して明確にします。
    """
    # 画像読み込み (OpenCVは [Height, Width, Channel] の順序)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"画像が見つかりません: {image_path}")

    # リサイズ (Width=64, Height=94)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # [重要] データの並び替えと正規化
    # 1. transpose(2, 0, 1): [H, W, C] -> [C, H, W] (PyTorchの仕様に合わせる)
    # 2. / 255.0: 画素値(0-255)を(0.0-1.0)に正規化 (学習時の条件に合わせる)
    img = img.transpose(2, 0, 1) / 255.0

    # 3. Tensor化 & バッチ次元の追加
    # モデルは一度に複数の画像を処理できるよう [Batch, C, H, W] の4次元を期待します。
    # unsqueeze(0) で先頭に次元を追加し、[1, 3, 94, 64] にします。
    img_tensor = torch.FloatTensor(img).unsqueeze(0)

    return img_tensor


# ==========================================
# 4. INFERENCE & DECODING (推論と後処理)
# ==========================================
def decode_prediction(output_tensor):
    """
    モデルの出力（確率）を、人間が読める文字列に変換する関数。
    """
    # 確率が最も高いインデックスを取得
    # dim=2 は「文字クラス(0~11)」の次元方向で最大値を探すことを意味します。
    # [1, 9, 12] -> [1, 9] (各文字位置での最強のインデックス)
    predict_indices = torch.argmax(output_tensor, dim=2)

    # バッチの先頭[0]を取り出し、インデックスを文字に変換して結合
    indices_list = predict_indices[0].tolist()  # Tensor -> Python List
    decoded_str = "".join([CHAR_LIST[i] for i in indices_list])

    return decoded_str


def run_inference(image_path):
    """
    推論プロセス全体を統括するメイン関数。
    """
    print(f"--- 推論開始: {image_path} ---")

    # 1. モデルの準備 (Load)
    # クラス数(12)を指定してインスタンス化
    model = SimpleLPNET(num_chars=len(CHAR_LIST))

    # 学習済み重み(weights)を読み込む
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        return

    # 推論モードに設定 (DropoutやBatchNormを固定)
    model.eval()

    # 2. 画像の前処理 (Pre-process)
    try:
        input_tensor = preprocess_image(image_path)
    except ValueError as e:
        print(e)
        return

    # 3. 推論実行 (Predict)
    # with torch.no_grad(): 勾配計算（学習用の記録）を無効化し、メモリと計算時間を節約
    with torch.no_grad():
        outputs = model(input_tensor)

    # 4. 結果の解釈 (Post-process)
    result_text = decode_prediction(outputs)

    print(f"認識結果: {result_text}")
    print("-------------------------")


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # テスト画像のパス
    target_image = r"C:\Users\user\Desktop\pycharm_projects\ALPR_projects\ocr_inputs\f63_p0.png"

    run_inference(target_image)