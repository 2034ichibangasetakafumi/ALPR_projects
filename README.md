# 自動車両番号認識(ALPR)システム開発

深層学習を用いた高精度な日本のナンバープレート認識（Automatic License Plate Recognition）システムを開発するプロジェクト。駐車場管理の省人化および入出庫車両ログ等の自動データベース化を目的とする
<img width="1024" height="687" alt="image" src="https://github.com/user-attachments/assets/8caf6b75-0d5a-42dc-8dc9-f91fa3a27bda" />
# デモ動画 (画像をクリックするとYouTubeで見れます)
[![デモ動画](https://img.youtube.com/vi/w5QBDchangs/maxresdefault.jpg)](https://www.youtube.com/watch?v=w5QBDchangs)
## 1. プロジェクト概要
### 1.1 開発の目的
OCR機能に深層学習を用い、エラーを極力回避した安定した動作によって駐車場管理業務の効率化を実現
<img width="579" height="132" alt="image" src="https://github.com/user-attachments/assets/3754792f-3c08-44ba-b87a-83735dda6dc9" />
<img width="275" height="142" alt="image" src="https://github.com/user-attachments/assets/ba074f3b-9c89-4aa4-8e86-4ea898749d61" />

### 1.2 背景
現状の目視による監視業務に対し、物体検出アルゴリズムとOCRを高度に統合することで処理を自動化<br>
 <img width="250" height="168" alt="image" src="https://github.com/user-attachments/assets/f38f71fd-1ef6-4f32-b515-8c32c424b3c1" />
 <img width="559" height="39" alt="image" src="https://github.com/user-attachments/assets/f45c3431-a59d-47a1-9594-a20d466fb5df" />

## 2. システム構成
### 2.1 ソフトウェアスタック
* **言語**: Python 3.10
* **主要ライブラリ**:
    * **OpenCV**: 画像の前処理、アフィン変換
    * **PyTorch**: 深層学習モデルの推論エンジン
    * **NumPy**: 多次元配列用並列計算エンジン

### 2.2 ハードウェア構成（目標）
* **撮影ユニット**: 2K解像度 IPカメラ（シャッタースピード固定、赤外線撮影）
* **演算ユニット**: 汎用PC（NVIDIA CUDAによるGPU処理）
* **ネットワーク**: POE接続による高品質インターネット回線

### 2.3 内包のプログラムについて(_anotatedは解説付きリファクタリングファイル) 
 * **Yolo_test04.py** : YOLOによるオブジェクトディテクションとナンバープレート切出し
 * **best.pt** : LPRNET推論用のウェイト（学習済みモデル）
 * **extract_frames.py** : 動画データを連番フレームに変換する際のフレームレート等も指定
 * **Ipnet.pth** : LPRNETモデルデータ
 * **numberPlateGenarateTest02.py** : 学習用のナンバープレート生成用
 * **predict.py** : 推論と画像認識
 * **split.py** : 生成したナンバープレートを上下に分ける（これにより各処理の精度を上げる）
 * **train_Ipnet.py** : トレーニング用
 * **yolov8n.pt** : YOLO用学習済みデータ（ウェイト）

## 3. 技術的アプローチ
システムは「抽出」「検出」「加工」「認識」の4フェーズで構成

1. **フレーム抽出 (Detection)**: OpenCVによりカメラデータをフレーム化し、後続の処理に最適化
2. **プレート検出 (Detection)**: **YOLOv8**を採用。車両全体を捉えた後、内部のナンバープレート領域をバウンディングボックスとして抽出
3. **前処理 (Pre-processing)**: 
    * 歪み補正のための**射影変換 (Perspective Transform)**
    * グレースケール変換、ガウシアンフィルタによるノイズ除去
    * 適応的閾値処理による文字強調
4. **文字認識 (Recognition)**: **LPRNet**を使用。日本特有の2段表示に対応するため、上段・下段個別の学習データ生成と日本語フォントを用いたデータ作成し使用

## 4. 法的問題
ナンバープレートの個人情報問題に対応するため、本プロジェクトはPythonプログラムにより自動生成されたナンバープレートを使用することとする

## 5. 今後の展望
* **学習データの拡充**: ナンバーの種類や時間帯など、多様な条件に対応するための継続的開発
* **機能拡充**: 通知機能、データベース連携、リアルタイム処理の実装
* **API連携**: 入出庫システムと連携し、駐車料金計算などの自動記録を実現

## 6.参考リンク

* **OpenCV**
    カメラデータのフレーム化や画像の前処理（アフィン変換、射影変換、ノイズ除去等）に使用<br> 
    [https://opencv.org/](https://opencv.org/)
* **Ultralytics YOLOv8 (GitHub)**
    車両およびナンバープレート領域を高精度に抽出するための最新物体検出モデル<br>
    [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **LPRNet (OCR)**
   ナンバープレートの地域名、分類番号、ひらがな、一連指定番号を個別に識別する推論エンジン<br>
    [https://github.com/sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) (PyTorch実装例)
