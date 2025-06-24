# task2
# 🧠 CIFAR-10 Image Classification using TensorFlow

## 📌 Internship Details

- **Company**: CodTech IT Solutions  
- **Name**: Jami Venkata Dhanush  
- **Intern ID**: CT06DL1263  
- **Domain**: Data Science  
- **Duration**: 6 Weeks  
- **Mentor**: Neela Santhosh  

---

## 📈 Project Overview

This project implements an end-to-end **Convolutional Neural Network (CNN)** using **TensorFlow** to classify images from the **CIFAR-10 dataset**. The model is trained to recognize 10 different object categories from colored 32x32 images. The deliverables include a trained model, accuracy/loss visualizations, and sample prediction comparisons.

---

## 🎯 Project Objective

> **Goal**: Implement a deep learning model for image classification using TensorFlow and visualize the training and prediction results.

---

## 🧾 Dataset Description: CIFAR-10

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is split into 50,000 training images and 10,000 test images.

### 🔟 Classes:
- ✈️ Airplane  
- 🚗 Automobile  
- 🐦 Bird  
- 🐱 Cat  
- 🦌 Deer  
- 🐶 Dog  
- 🐸 Frog  
- 🐴 Horse  
- 🚢 Ship  
- 🚚 Truck  

---

## 🏗️ Model Architecture

![Image](https://github.com/user-attachments/assets/700e1b9b-4b88-4d90-b29a-f1b1b3b752c5)

```python
Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

