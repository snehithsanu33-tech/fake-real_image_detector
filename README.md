# Fake / Real Image Detection System

A deep learning-based web application that classifies an uploaded image as **Fake** or **Real** using a trained **MobileNetV2** model.

## Project Overview

The Fake / Real Image Detection System is designed to classify images using deep learning and transfer learning. The application provides a simple web interface where users can upload an image and receive a prediction from the trained model.

The trained model is integrated with a Flask backend and connected to a web-based frontend.

## Features

- Upload an image through a web interface
- Automatic image preprocessing
- Deep learning-based image classification
- Fake / Real prediction
- Simple and user-friendly interface
- Flask-based backend
- MobileNetV2 transfer learning model

## Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **MobileNetV2**
- **Flask**
- **HTML**
- **CSS**
- **JavaScript**
- **Pillow**
- **NumPy**

## Machine Learning Model

The project uses **MobileNetV2** with transfer learning for binary image classification.

The model architecture consists of:

- MobileNetV2 pretrained on ImageNet
- Global Average Pooling
- Dense layer with 128 neurons
- Dropout layer
- Sigmoid output layer for binary classification

### Model Workflow

```text
User Uploads Image
        ↓
Image Preprocessing
        ↓
MobileNetV2 Model
        ↓
Binary Classification
        ↓
Fake / Real Prediction
        ↓
Result Displayed on Website