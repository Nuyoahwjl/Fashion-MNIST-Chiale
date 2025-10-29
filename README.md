# 🧠 Fashion-MNIST Classification with Hand-Crafted CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![Accuracy](https://img.shields.io/badge/Max_Accuracy-86.12%25-brightgreen.svg)](#results) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-orange.svg)](https://github.com/zalandoresearch/fashion-mnist)

A pure numpy-based convolutional neural network (CNN) implementation for classifying the Fashion-MNIST dataset, without using any deep learning frameworks. This project demonstrates the core principles of CNNs, including convolution, pooling, batch normalization, and backpropagation.


## 📋 Project Overview
Fashion-MNIST is a popular dataset consisting of 60,000 training images and 10,000 test images of 10 fashion categories. This project implements a LeNet-inspired CNN from scratch using only NumPy, achieving ~86% test accuracy after 5000 training steps.


## 🔍 Key Features
- **Framework-Free**: Entirely built with NumPy, no TensorFlow/PyTorch or other ML frameworks.
- **Complete Pipeline**: Includes data loading, model implementation, training, prediction, and visualization.
- **Core CNN Components**: Implements Conv2D, BatchNorm, ReLU, MaxPool2D and FullyConnected layers.
- **Training Utilities**: Logging, model saving/loading, and performance visualization.


## 🧱 Model Architecture (LeNet Variant)
The model follows a modified LeNet architecture with batch normalization for better performance:
```
Layer (type)         Output Shape         Param #
==================================================
Conv2D               (6, 28, 28)          156
BatchNorm            (6, 28, 28)          12
ReLU                 (6, 28, 28)          0
MaxPool2D            (6, 14, 14)          0
Conv2D               (16, 10, 10)         2416
BatchNorm            (16, 10, 10)         32
ReLU                 (16, 10, 10)         0
MaxPool2D            (16, 5, 5)           0
FullyConnected       (120,)               48120
ReLU                 (120,)               0
FullyConnected       (84,)                10164
ReLU                 (84,)                0
FullyConnected       (10,)                850
==================================================
Total params: 61750
```

![Model Architecture](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/1.png)
![Layer Visualization](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/2.png)


## ⚙️ Training Hyperparameters
| Parameter         | Value    |
|-------------------|----------|
| max_steps         | 5000     |
| batch_size        | 64       |
| learning_rate     | 0.0005   |


## 📊 Results

### Loss & Accuracy Curves
Training curves for 1000 steps and 5000 steps (smoothed for clarity):

| *1000 Steps*     | *5000 Steps*     |
|:----------------:|:----------------:|
| ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/5.png) | ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/3.png) |
| ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/6.png) | ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/4.png) |


### Test Accuracy by Training Steps
| Model               | Accuracy |
|---------------------|----------|
| model_step1000.npz  | 0.8016   |
| model_step2000.npz  | 0.8264   |
| model_step3000.npz  | 0.8462   |
| model_step4000.npz  | 0.8534   |
| model_step5000.npz  | 0.8612   |


### Confusion Matrix
Confusion matrix for the final model (5000 steps) showing class-wise performance:

![Confusion Matrix](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/7.png)


## 🚀 Usage

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn prettytable
```

### Training
Run the training script to train the model from scratch:
```bash
python train.py
```
- Trained models are saved in the `models/` directory every 1000 steps.
- Training logs (loss, accuracy) are saved in the `logs/` directory.


### Prediction & Evaluation
Run the prediction script to evaluate trained models on test data:
```bash
python predict.py
```
- Generates confusion matrices and prints test accuracy for each saved model.


### Visualization
Plot training loss and accuracy curves using:
```bash
python plot.py
```


## 📁 Project Structure
```
Fashion-MNIST-Chiale/
├── cnn_model.py       # CNN model implementation (layers, forward/backward pass)
├── train.py           # Training pipeline
├── predict.py         # Prediction and evaluation
├── plot.py            # Visualization of training curves
├── data_loader.py     # Fashion-MNIST data loading
├── data/              # Dataset directory (Fashion-MNIST files)
├── models/            # Saved model checkpoints
├── logs/              # Training logs (loss, accuracy)
└── img/               # Visualization images
```


## 🙌 Acknowledgements
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) for providing the benchmark data.
- Inspired by the LeNet-5 architecture for image classification tasks.
