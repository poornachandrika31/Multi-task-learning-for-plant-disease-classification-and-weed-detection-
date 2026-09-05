
#  A Multi-Task Deep Learning Framework for Plant Disease Classification and Weed Detection

## 🔗 Project Resources

## 🎥 90-Second Demo

▶️ **[Watch the 90-Second Demo](./project_demo(1).mp4)**

## 📄 Research Paper

📑 **[Read the Research Paper](./RESEARCH PAPER_IPF25.pdf)**

## 📌 Project Overview

This project presents a **router-based multi-task deep learning framework** for intelligent agricultural analysis by integrating **plant disease classification** and **weed detection** into a unified computer vision pipeline. Instead of relying on separate systems for different agricultural tasks, the framework dynamically routes an input image to the most appropriate deep learning model using a lightweight routing network.

The framework combines **EfficientNet-B0** as a routing network, **MobileNetV3-Large** for plant disease classification, and **YOLOv5s** for real-time weed detection. This architecture reduces redundant computation while maintaining high prediction accuracy across multiple agricultural vision tasks.

---

# 📌 Motivation

Traditional agricultural AI systems deploy separate deep learning models for different tasks such as disease diagnosis and weed identification. Running multiple models for every image increases computational cost, inference time, and resource consumption.

This project addresses these challenges by introducing a **router-based inference pipeline** capable of automatically selecting the appropriate model based on the input image, making the framework more efficient and scalable for precision agriculture applications.

---

# 📌 Objectives

* Develop a unified multi-task computer vision framework.
* Classify plant diseases using lightweight CNN architectures.
* Detect weeds in agricultural fields using object detection.
* Dynamically route images to task-specific models.
* Reduce unnecessary inference while maintaining high accuracy.
* Build modular preprocessing, training, and evaluation pipelines.

---

# 📌 Datasets

### 🌿 Plant Disease Dataset

* PlantVillage Dataset
* Multiple crop species
* Healthy and diseased plant leaves
* Image Classification

### 🌾 Weed Detection Dataset

* Agricultural field images
* Bounding box annotations
* Multiple weed species
* Object Detection

---

# 📌 System Architecture

```
                    Input Image
                         │
                         ▼
               EfficientNet-B0 Router
                  /               \
                 /                 \
                ▼                   ▼
 MobileNetV3-Large          YOLOv5s Detector
 Disease Classification      Weed Detection
                \                 /
                 \               /
                  ▼             ▼
             Unified Prediction Output
```

---

# 📌 Methodology

The framework consists of three major components.

## 1. Image Preprocessing

Input images undergo

* Image resizing
* Normalization
* Data augmentation
* Quality enhancement
* Dataset balancing

using OpenCV and deep learning preprocessing pipelines.

---

## 2. Routing Network

A lightweight **EfficientNet-B0** model serves as a routing network.

Instead of performing every task on every image, the router first predicts the image category and forwards it to the corresponding specialized model.

This reduces

* inference latency
* computational cost
* unnecessary model execution

---

## 3. Plant Disease Classification

Plant disease images are routed to **MobileNetV3-Large**.

Transfer learning is employed to classify healthy and diseased crop leaves across multiple disease categories.

Features include

* Transfer Learning
* Fine-tuning
* Lightweight architecture
* Efficient inference

---

## 4. Weed Detection

Agricultural field images are routed to **YOLOv5s**.

The detector performs

* Weed localization
* Bounding box prediction
* Confidence estimation
* Multi-object detection

making it suitable for precision agriculture.

---

# 📌 Tech Stack

### Programming

* Python

### Deep Learning

* TensorFlow
* PyTorch

### Computer Vision

* OpenCV

### Models

* EfficientNet-B0
* MobileNetV3-Large
* YOLOv5s

### Libraries

* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# 📌 Pipeline

```
Dataset Collection
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
EfficientNet-B0 Router
       / \
      /   \
     ▼     ▼
MobileNet YOLOv5
     │      │
     ▼      ▼
Prediction Fusion
        │
        ▼
Final Output
```

---

# 📌 Evaluation Metrics

The framework was evaluated using

### Classification

* Accuracy
* Precision
* Recall
* F1-score

### Detection

* Precision
* Recall
* mAP
* IoU
* Bounding Box Localization

---

# 📌 Key Features

* Multi-task deep learning framework
* Dynamic task routing
* Transfer learning
* Lightweight CNN models
* Real-time inference
* Modular architecture
* Scalable deployment pipeline
* Unified agricultural vision system

---

# 📌 Results

The proposed framework successfully

* Classified plant diseases with high accuracy using MobileNetV3-Large.
* Detected weeds using YOLOv5s with robust localization performance.
* Reduced redundant inference through EfficientNet-B0 based routing.
* Provided a unified and scalable solution for agricultural computer vision tasks.

---

# 📌 Future Work

Potential improvements include

* Vision Transformers (ViT)
* Multi-modal agricultural analysis
* Edge deployment on NVIDIA Jetson and Raspberry Pi
* Drone-based crop monitoring
* Real-time video analytics
* Transformer-based object detection (DETR)
* Large Vision-Language Models for agricultural assistance

---

# 📌 Repository Structure

```
📦 MultiTask-Agriculture-AI
│
├── datasets/
│   ├── PlantVillage/
│   ├── WeedDataset/
│
├── preprocessing/
│
├── models/
│   ├── EfficientNetB0_Router/
│   ├── MobileNetV3/
│   ├── YOLOv5/
│
├── training/
│
├── inference/
│
├── evaluation/
│
├── notebooks/
│
├── results/
│
├── app.py
│
├── requirements.txt

