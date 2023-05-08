# Synthetic Image Segmentation with Domain Adaptation

This repository contains the code and resources for the research project on Synthetic Image Segmentation. The goal of the project is to perform image segmentation on simulated data and use domain adaptation techniques to generalize the model to real-world data.

## Overview

Image segmentation is a critical task in computer vision, where the goal is to partition an image into semantically meaningful segments, assigning each pixel to a specific object or background. Training deep learning models for image segmentation typically requires large amounts of labeled data. However, acquiring labeled data for real-world images can be expensive and time-consuming.

Synthetic data, generated through simulations, offers an alternative approach. By using synthetic data, we can generate large amounts of labeled data with minimal effort. However, models trained on synthetic data often do not generalize well to real-world data due to differences in the distribution of the data, known as the domain gap.

In this research project, we focus on training image segmentation models on synthetic data and leveraging domain adaptation techniques to bridge the domain gap and improve performance on real-world data.

## Project Structure

The project is divided into two main components:

1. **Unity Simulation**: The Unity-based simulation generates synthetic images, along with various modalities such as object segmentation, normals, depth, and outlines. This data is used for training the image segmentation models.

2. **Python Data Loader and Model Training**: The Python part of the pipeline processes the data from the Unity simulation, feeding it into a data loader that is compatible with PyTorch Lightning. The data loader is then used to train image segmentation models, employing domain adaptation techniques to improve generalization to real-world data.
