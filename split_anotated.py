import os
import random
import shutil

def split_dataset(src_dir="dataset_fixed", train_ratio=0.8):
    # フォルダ作成
    train_dir = "train_data"
    val_dir = "val_data"
    for d in [train_dir, val_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # labels.txtを読み込み
    with open(f"{src_dir}/labels.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # シャッフルして分割
    random.shuffle(lines)
    split_point = int(len(lines) * train_ratio)
    train_lines = lines[:split_point]
    val_lines = lines[split_point:]

    # ファイル移動とラベル作成
    for data, d_name in [(train_lines, train_dir), (val_lines, val_dir)]:
        with open(f"{d_name}/labels.txt", "w", encoding="utf-8") as f:
            for line in data:
                img_name, label = line.strip().split(" ")
                shutil.copy(f"{src_dir}/{img_name}", f"{d_name}/{img_name}")
                f.write(f"{img_name} {label}\n")

    print(f"分割完了: 学習用 {len(train_lines)}枚 / 検証用 {len(val_lines)}枚")

split_dataset()