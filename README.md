# Fashion-MNIST Classification with Hand-Crafted CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy Only](https://img.shields.io/badge/NumPy-only-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Max Accuracy](https://img.shields.io/badge/Max_Accuracy-86.12%25-brightgreen.svg)](#-results)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-FFCB2B.svg?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/Nuyoahwjl/Fashion-MNIST-Chiale/blob/main/demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-0b7285.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Nuyoahwjl/Fashion-MNIST-Chiale?style=social)](https://github.com/Nuyoahwjl/Fashion-MNIST-Chiale/stargazers)

A pure NumPy implementation of a convolutional neural network (CNN) for classifying the Fashion‑MNIST dataset—no deep learning frameworks required. This project demonstrates the core principles behind CNNs by building layers, forward/backward passes, and training loops from scratch.

## 📋 Project Overview
Fashion‑MNIST contains 60,000 training images and 10,000 test images across 10 fashion categories. This repository implements a LeNet‑inspired CNN using only NumPy, covering data loading, training, evaluation, and visualization to help you understand CNN fundamentals end‑to‑end.

## 🔍 Key Features
- 🧠 Framework‑Free: built entirely with NumPy—no TensorFlow/PyTorch.
- 🧪 Complete Pipeline: data loading, training, evaluation, visualization.
- 🧱 Core Layers: Conv2D, BatchNorm, ReLU, MaxPool2D, FullyConnected.
- 🗂️ Utilities: logging, checkpointing, and performance plots.
- 🧑‍🏫 Interactive: Jupyter notebook demo for exploration.

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
```
| Parameter         | Value    |
|-------------------|----------|
| max_steps         | 5000     |
| batch_size        | 64       |
| learning_rate     | 0.0005   |
```

## 📊 Results

### Loss & Accuracy Curves
Training curves for 1000 steps and 5000 steps (smoothed for clarity):

| *1000 Steps*     | *5000 Steps*     |
|:----------------:|:----------------:|
| ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/5.png) | ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/3.png) |
| ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/6.png) | ![](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/4.png) |

### Test Accuracy by Training Steps
```
| Model               | Accuracy |
|---------------------|----------|
| model_step1000.npz  | 0.8016   |
| model_step2000.npz  | 0.8264   |
| model_step3000.npz  | 0.8462   |
| model_step4000.npz  | 0.8534   |
| model_step5000.npz  | 0.8612   |
```

### Confusion Matrix
Confusion matrix for the final model (5000 steps) showing class-wise performance:

![Confusion Matrix](https://cdn.jsdelivr.net/gh/Nuyoahwjl/Fashion-MNIST-Chiale/img/7.png)


## 🚀 Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install numpy matplotlib scikit-learn prettytable pandas
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
### Interactive Demo
Explore the interactive Jupyter notebook for a hands-on demonstration:
```bash
jupyter notebook demo.ipynb
```

## 📁 Project Structure
```
Fashion-MNIST-Chiale/
├── cnn_model.py       # CNN model implementation (layers, forward/backward pass)
├── train.py           # Training pipeline
├── predict.py         # Prediction and evaluation
├── plot.py            # Visualization of training curves
├── data_loader.py     # Fashion-MNIST data loading
├── demo.ipynb         # Interactive Jupyter notebook demo
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
├── data/              # Dataset directory (Fashion-MNIST files)
├── models/            # Saved model checkpoints
├── logs/              # Training logs (loss, accuracy)
└── img/               # Visualization images
```

## 🙌 Acknowledgements
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) for providing the benchmark data.
- Inspired by the LeNet-5 architecture for image classification tasks.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
Github: [@Nuyoahwjl](https://github.com/Nuyoahwjl)

## ⭐ Show Your Support
Give a ⭐️ if this project helped you learn about CNNs or deep learning fundamentals!

