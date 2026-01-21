import cv2
import numpy as np
import random
import os  # ← これが必須です
from PIL import Image, ImageDraw, ImageFont

# 辞書定義
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
    'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と',
    'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ',
    'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り',
    'る', 'れ', 'ろ', 'わ', 'を',
    '品川', '横浜', '足立', '練馬', '大宮', '多摩',
    '佐賀', '久留米', '福岡', '熊本', '大分', '長崎', '佐世保', '宮崎', '鹿児島', '沖縄', '那覇', '北九州',
    '尾張小牧',
]


def generate_perfect_split(count=100):
    output_dir = "dataset"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. ラベルファイルをオープン
    with open(f"{output_dir}/labels.txt", "w", encoding="utf-8") as label_file:

        # 2. 白黒設定を定義
        bg_color = (255, 255, 255)
        text_color = (0, 0, 0)

        font_path = "C:/Windows/Fonts/msgothic.ttc"
        f_upper = ImageFont.truetype(font_path, 80)
        f_num = ImageFont.truetype(font_path, 90)

        AREAS = [c for c in CHARS if len(c) > 1]
        KANA_LIST = CHARS[10:55]

        for i in range(count):
            area = random.choice(AREAS)
            class_num = random.randint(300, 399)
            kana = random.choice(KANA_LIST)
            num_main = f"{random.randint(0, 9)}{random.randint(0, 9)}-{random.randint(0, 9)}{random.randint(0, 9)}"

            # --- ここから上段の座標・サイズ調整 ---
            length = len(area)
            if length == 4:
                start_x = 10  # 尾張小牧用
                f_size = 70
            elif length == 3:
                start_x = 30  # 北九州・久留米など
                f_size = 80
            else:  # 品川・横浜など
                start_x = 80
                f_size = 80

            # その都度フォントオブジェクトを作成
            f_upper_dynamic = ImageFont.truetype(font_path, f_size)
            # -----------------------------------

            # --- 上段生成 ---
            img_up = Image.new("RGB", (440, 110), bg_color)
            draw_up = ImageDraw.Draw(img_up)
            txt_up = f"{area}{class_num}"
            draw_up.text((start_x, 10), f"{area} {class_num}", font=f_upper_dynamic, fill=text_color)

            # ↓ ここを追加しないと上段が保存されません
            up_name = f"{i}_up.png"
            cv2.imwrite(f"{output_dir}/{up_name}", cv2.resize(np.array(img_up), (94, 24)))
            label_file.write(f"{up_name} {txt_up}\n")

            # --- 下段生成 --- (以下は提示されたコードでOK)            # --- 下段生成 ---
            img_low = Image.new("RGB", (440, 110), bg_color)  # 変数を使用
            draw_low = ImageDraw.Draw(img_low)
            txt_low = f"{kana}{num_main}".replace("-", "")
            draw_low.text((20, 10), f"{kana}  {num_main}", font=f_num, fill=text_color)  # 変数を使用

            low_name = f"{i}_low.png"
            cv2.imwrite(f"{output_dir}/{low_name}", cv2.resize(np.array(img_low), (94, 24)))
            label_file.write(f"{low_name} {txt_low}\n")

    print(f"{count}セット生成完了（labels.txt作成済）")
generate_perfect_split(count=100)