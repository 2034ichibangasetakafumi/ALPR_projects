
[🇯🇵 日本語版 → README_ja.md](README_ja.md)

# Automatic License Plate Recognition (ALPR) System Development

This project aims to develop a **high-accuracy Japanese Automatic License Plate Recognition (ALPR) system** using deep learning techniques.  
The main objective is to **automate parking management operations** and **build a vehicle entry/exit logging system**, reducing human workload and improving operational efficiency.

<img width="1024" height="687" alt="image" src="https://github.com/user-attachments/assets/8caf6b75-0d5a-42dc-8dc9-f91fa3a27bda" />

# Demo Video (Click the image to watch on YouTube)
[[![Demo Video](https://img.youtube.com/vi/w5QBDchangs/maxresdefault.jpg)](https://www.youtube.com/watch?v=w5QBDchangs)](https://youtu.be/KVvExKfSeIc)

---

## 1. Project Overview

### 1.1 Purpose of Development
By applying deep learning–based OCR and robust preprocessing techniques, this project aims to achieve **stable and highly accurate license plate recognition**, enabling efficient automation of parking facility operations.

<img width="579" height="132" alt="image" src="https://github.com/user-attachments/assets/3754792f-3c08-44ba-b87a-83735dda6dc9" />
<img width="275" height="142" alt="image" src="https://github.com/user-attachments/assets/ba074f3b-9c89-4aa4-8e86-4ea898749d61" />

---

### 1.2 Background
Current parking management often relies on **manual visual inspection**, which is labor-intensive and error-prone.  
By integrating **object detection algorithms and OCR technologies**, the system automates vehicle identification and logging.

<img width="250" height="168" alt="image" src="https://github.com/user-attachments/assets/f38f71fd-1ef6-4f32-b515-8c32c424b3c1" />
<img width="559" height="39" alt="image" src="https://github.com/user-attachments/assets/f45c3431-a59d-47a1-9594-a20d466fb5df" />

---

## 2. System Architecture

### 2.1 Software Stack
- **Language**: Python 3.10
- **Main Libraries**:
  - **OpenCV**: Image preprocessing and affine transformations
  - **PyTorch**: Deep learning inference engine
  - **NumPy**: High-performance numerical computing

---

### 2.2 Hardware Configuration (Target Environment)
- **Imaging Unit**: 2K resolution IP camera (fixed shutter speed, infrared support)
- **Processing Unit**: General-purpose PC with NVIDIA GPU (CUDA acceleration)
- **Network**: High-speed PoE-based IP network

---

### 2.3 Included Programs  
(*Files suffixed with `_annotated` contain refactored versions with detailed explanations*)

- **Yolo_test04.py**: Object detection using YOLO and license plate cropping
- **best.pt**: Pretrained YOLO model weights
- **extract_frames.py**: Extracts video frames with configurable frame rate
- **Ipnet.pth**: LPRNet trained model weights
- **numberPlateGenarateTest02.py**: Synthetic license plate generator for training
- **predict.py**: Inference pipeline for detection and OCR
- **split.py**: Splits license plates into upper and lower sections for accuracy improvement
- **train_Ipnet.py**: Training script for LPRNet
- **yolov8n.pt**: YOLOv8 pretrained weights

---

## 3. Technical Approach

The system is structured into four major processing phases:

1. **Frame Extraction (Preprocessing)**  
   Video streams are converted into frames using OpenCV and optimized for downstream processing.

2. **License Plate Detection (Detection)**  
   **YOLOv8** is used to detect vehicles and localize license plate regions via bounding boxes.

3. **Image Preprocessing (Pre-processing)**  
   - Perspective transformation for geometric correction  
   - Grayscale conversion and Gaussian filtering for noise reduction  
   - Adaptive thresholding for character enhancement  

4. **Character Recognition (Recognition)**  
   **LPRNet** is employed for OCR. To support the **two-row layout unique to Japanese plates**, training data is generated separately for upper and lower sections using Japanese fonts.

---

## 4. Legal and Privacy Considerations

To address privacy concerns related to personal data, this project exclusively uses **synthetically generated license plates** created via Python-based data generation scripts.

---

## 5. Future Work

- **Dataset Expansion**: Improve robustness under varying lighting conditions, weather, and plate styles
- **Feature Enhancements**: Real-time processing, notification systems, and database integration
- **API Integration**: Link with parking management systems to automate billing and access control

---

## 6. References

- **OpenCV**  
  Used for frame extraction, affine transformation, perspective correction, and noise filtering  
  https://opencv.org/

- **Ultralytics YOLOv8 (GitHub)**  
  State-of-the-art object detection model for vehicle and license plate localization  
  https://github.com/ultralytics/ultralytics

- **LPRNet (OCR Engine)**  
  OCR engine specialized for license plate recognition  
  https://github.com/sirius-ai/LPRNet_Pytorch
