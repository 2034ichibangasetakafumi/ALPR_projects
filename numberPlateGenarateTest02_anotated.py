import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIGURATION (設定エリア)
# ==========================================
# 保存先設定
OUTPUT_DIR = "dataset_fixed"
FONT_PATH = "C:/Windows/Fonts/msgothic.ttc"

# 生成設定
GENERATE_COUNT = 500  # 作成するセット数
TARGET_SIZE = (94, 24)  # 【重要】LPRNetモデルへの入力サイズ（幅94, 高さ24）

# 固定するナンバープレート情報
# 今回は特定のナンバーを学習させるためのデータセットを作成します
PLATE_INFO = {
    "up": {
        "text": "佐賀 581",  # 描画する文字（空白含む）
        "label": "佐賀581",  # 教師データとしての正解ラベル
        "font_size": 80
    },
    "low": {
        "text": "せ  18-32",
        "label": "せ1832",
        "font_size": 90
    }
}


def create_synthetic_image(text, font_path, font_size, bg_val, txt_val):
    """
    PILを使用して、指定されたテキストを描画した画像を生成する関数。

    【なぜ関数化するのか？】
    - **役割の分離**: OpenCVは「画像処理」は得意ですが、「文字描画（特に日本語）」は苦手です。
      ここではPIL(Pillow)を使って文字を描画する工程を独立させました。
    """
    # フォントの読み込み
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: フォントが見つかりません -> {font_path}")
        return None

    # 画像キャンバスの作成 (RGBモード, サイズ440x110)
    # 背景色 (bg_val) をランダムにすることで、照明変化への頑健性を高めます。
    img = Image.new("RGB", (440, 110), (bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)

    # 文字の描画
    # 座標(20, 10)などは経験的な配置位置です
    draw.text((20, 10), text, font=font, fill=(txt_val, txt_val, txt_val))

    return img


def apply_augmentation_and_resize(pil_img):
    """
    生成された画像に対して、AIモデル用の前処理とノイズ付加を行う関数。

    【なぜ関数化するのか？】
    - **データパイプラインの明確化**: ここは「人間が見るための画像」を「AIが読むためのデータ」に変換する工程です。
    """
    # 1. データ形式の変換 (PIL -> OpenCV)
    # PIL(RGB) から OpenCV(BGR) へ変換します。AIフレームワークの多くはOpenCV形式での読み込みを前提としています。
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 2. リサイズ (Preprocessing)
    # LPRNetなどの軽量モデルは、入力サイズが (94, 24) など非常に小さい解像度で固定されていることが多いです。
    img_cv = cv2.resize(img_cv, TARGET_SIZE)

    # 3. ノイズ付加 (Data Augmentation)
    # 50%の確率で「ガウスぼかし」を加えます。
    # 綺麗なデジタル画像だけでなく、ピンボケ画像も学習させることで、実環境での認識精度を上げます。
    if random.random() > 0.5:
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)

    return img_cv


def generate_fixed_plate_dataset(count):
    """
    メイン処理: ループを回してデータセットを生成し、ファイルに保存する。
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ラベルファイル（正解データリスト）を開く
    # フォーマット: [ファイル名] [正解ラベル]
    label_file_path = os.path.join(OUTPUT_DIR, "labels.txt")

    with open(label_file_path, "w", encoding="utf-8") as f:
        print(f"生成開始: {count}セット作成します...")

        for i in range(count):
            # ランダムな色味の決定 (Domain Randomization)
            # 毎回微妙に色を変えることで、過学習（特定の背景色でしか認識できなくなる現象）を防ぎます。
            bg_val = random.randint(230, 255)  # 背景は白に近いグレー
            txt_val = random.randint(0, 40)  # 文字は黒に近いグレー

            # 上段(地域・分類番号) と 下段(ひらがな・一連番号) をそれぞれ作成
            for part_name, info in PLATE_INFO.items():

                # 1. 画像生成 (PIL)
                img_pil = create_synthetic_image(
                    info["text"], FONT_PATH, info["font_size"], bg_val, txt_val
                )
                if img_pil is None: continue

                # 2. 加工・リサイズ (OpenCV)
                final_img = apply_augmentation_and_resize(img_pil)

                # 3. 保存 (Output)
                file_name = f"{i}_{part_name}.png"
                save_path = os.path.join(OUTPUT_DIR, file_name)

                cv2.imwrite(save_path, final_img)

                # ラベル書き込み
                # 学習時に「この画像には何が書いてあるか」を教えるためのファイルです。
                f.write(f"{file_name} {info['label']}\n")

    print(f"完了: 特定ナンバーのデータを{count}セット生成しました。")
    print(f"保存先: {OUTPUT_DIR}")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    generate_fixed_plate_dataset(GENERATE_COUNT)