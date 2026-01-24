import cv2
import os

# ==========================================
# CONFIGURATION (設定エリア)
# ==========================================
# Input: 動画ファイルのパス
VIDEO_PATH = r"C:\Users\user\Desktop\1768150033193_0.mp4"

# Output: 切り出した画像の保存先フォルダ
OUT_DIR = "frames"

# Processing Settings: 処理範囲と密度の設定
START_FRAME = 0  # 開始フレーム番号
END_FRAME = 150  # 終了フレーム番号
SKIP_INTERVAL = 1  # 保存間隔 (1 = 全フレーム保存, 10 = 10フレームごとに1枚保存)


def extract_frames_from_video(video_path, output_dir, start_frame, end_frame, skip_interval):
    """
    動画からフレーム（画像）を連続で切り出す関数。

    【なぜ関数化するのか？】
    1. **ロジックの分離**: 「設定値（どこ）」と「処理（どうやるか）」を分けることで、コードが読みやすくなります。
    2. **再利用性**: 将来、別の動画を処理したくなった場合、設定値を変えてこの関数を呼ぶだけで済みます。
    3. **スコープの限定**: `cap` や `frame` などの変数がグローバルに残らないため、予期せぬバグを防げます。
    """

    # ------------------------------------------
    # 1. INITIALIZATION (準備フェーズ)
    # ------------------------------------------

    # 保存先ディレクトリが存在しない場合は作成する
    # ※ YOLOなどの学習を行う際、データセットを整理するためにフォルダ分けは必須です。
    os.makedirs(output_dir, exist_ok=True)

    # OpenCV (cv2) を使って動画ファイルを読み込む準備
    # Context: YOLOやOCRは「動画」を直接見るのではなく、「静止画」の連続として処理します。
    # このステップは、AIにデータを渡すための「入り口 (Input)」となります。
    cap = cv2.VideoCapture(video_path)

    # 動画が正しく開けたか確認（エラーハンドリング）
    if not cap.isOpened():
        print(f"Error: 動画を開けませんでした -> {video_path}")
        return

    # 指定した開始位置（START_FRAME）まで「シーク（移動）」する
    # これにより、不要な最初のフレームを読み込む時間を短縮できます。
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame_index = start_frame  # 現在処理中のフレーム番号
    saved_count = 0  # 保存した画像の枚数

    print(f"処理開始: フレーム {start_frame} から {end_frame} まで抽出します...")

    # ------------------------------------------
    # 2. PROCESSING LOOP (処理ループ)
    # ------------------------------------------
    while cap.isOpened():
        # 【Input】 次のフレームを1枚読み込む
        # ret (bool): 読み込み成功ならTrue、失敗（動画の終わりなど）ならFalse
        # frame (array): 画像データそのもの（画素値の配列）
        ret, frame = cap.read()

        # ループ終了条件:
        # 1. 動画の末尾に到達した (retがFalse)
        # 2. 設定した終了フレーム (END_FRAME) を超えた
        if not ret or current_frame_index > end_frame:
            break

        # 【Processing】 指定した間隔（SKIP_INTERVAL）ごとに保存処理を行う
        # すべてのフレームを保存するとデータが多すぎる場合（似た画像ばかりになる）、
        # ここで間引きを行うことで、学習データの「質」と「量」を調整します。
        if (current_frame_index - start_frame) % skip_interval == 0:
            # 保存ファイル名の作成 (例: frames/frame_00001.jpg)
            # ※ 05d は「5桁のゼロ埋め」という意味です（整列しやすくするため）
            output_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")

            # 【Output】 画像をディスクに書き出す
            # Context (LPRNet/OCR): ここで保存されたJPEG画像が、後の工程で
            # 「ナンバープレート認識 (LPRNet)」や「文字認識 (OCR)」の入力データとして使われます。
            cv2.imwrite(output_filename, frame)

            saved_count += 1

        # 次のフレームへカウントを進める
        current_frame_index += 1

    # ------------------------------------------
    # 3. CLEANUP (終了処理)
    # ------------------------------------------
    # メモリ解放: 動画ファイルを閉じる
    # これを忘れると、他のソフトで動画が開けなくなることがあります。
    cap.release()
    print(f"完了: 合計 {saved_count} 枚の画像を '{output_dir}' に保存しました。")


# ==========================================
# ENTRY POINT (実行ブロック)
# ==========================================
if __name__ == "__main__":
    # 定数を使って関数を実行
    extract_frames_from_video(
        VIDEO_PATH,
        OUT_DIR,
        START_FRAME,
        END_FRAME,
        SKIP_INTERVAL
    )