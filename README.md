# MNIST-ANN-CNN-Study

This repository contains practice exercises and experiments on **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** for handwritten digit recognition using the **MNIST dataset**. The project includes model building, training, evaluation, visualization, confusion matrix analysis, model saving/loading, and basic CNN operations such as max pooling.

## Overview

The main goal of this repository is to study and compare deep learning approaches for image classification, especially digit recognition from grayscale 28x28 images. The code covers both fully connected neural networks and convolutional neural networks, along with supporting experiments for data preprocessing, evaluation, and model reuse.

## Main Contents

### 1. ANN-based Digit Classification
The repository includes multiple ANN experiments for MNIST-style digit classification:
- basic feedforward neural networks
- different hidden layer sizes
- training and validation split
- prediction visualization
- saved model loading and reuse

### 2. CNN-based Digit Classification
CNN models are implemented to improve image classification performance on MNIST:
- convolution layers
- max pooling
- dropout
- flatten and dense layers
- training with different optimizers such as **Adadelta** and **Adam**

### 3. Model Evaluation
Several files focus on evaluating model performance through:
- validation accuracy
- confusion matrix
- wrong prediction visualization
- comparison of predicted and true labels

### 4. Supporting Experiments
The repository also includes smaller learning exercises such as:
- manual ANN forward prediction
- dataset loading
- max pooling demonstration
- train/validation splitting
- model save and load workflow

## Files Included

Example scripts in this repository include:
- ANN training and testing scripts
- CNN training with Adadelta
- CNN training with Adam
- confusion matrix analysis
- model save/load scripts
- max pooling example
- MNIST data handling utilities

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Learning Objectives

- Understand the structure of ANN and CNN models
- Practice training deep learning models on image datasets
- Evaluate classification performance using confusion matrices
- Visualize predictions and misclassified samples
- Learn how to save and reload trained models
- Explore basic CNN operations such as convolution and pooling

## How to Run

Install the required libraries:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
