import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont


def generate_fixed_plate(count=500):
    output_dir = "dataset_fixed"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    font_path = "C:/Windows/Fonts/msgothic.ttc"
    f_upper = ImageFont.truetype(font_path, 80)
    f_num = ImageFont.truetype(font_path, 90)

    # 固定する情報
    area_text = "佐賀 581"
    label_up = "佐賀581"
    main_text = "せ  18-32"
    label_low = "せ1832"

    with open(f"{output_dir}/labels.txt", "w", encoding="utf-8") as f:
        for i in range(count):
            # わずかな色のゆらぎ（ノイズ）
            bg_val = random.randint(230, 255)
            txt_val = random.randint(0, 40)

            for part in ["up", "low"]:
                img = Image.new("RGB", (440, 110), (bg_val, bg_val, bg_val))
                draw = ImageDraw.Draw(img)

                if part == "up":
                    draw.text((30, 10), area_text, font=f_upper, fill=(txt_val, txt_val, txt_val))
                    txt_label = label_up
                else:
                    draw.text((20, 10), main_text, font=f_num, fill=(txt_val, txt_val, txt_val))
                    txt_label = label_low

                # OpenCV形式に変換してリサイズ
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img_cv = cv2.resize(img_cv, (94, 24))

                # わずかなガウスぼかし
                if random.random() > 0.5:
                    img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)

                file_name = f"{i}_{part}.png"
                cv2.imwrite(f"{output_dir}/{file_name}", img_cv)
                f.write(f"{file_name} {txt_label}\n")

    print(f"特定ナンバー『佐賀581せ18-32』のデータを{count}セット生成しました。")


generate_fixed_plate(500)