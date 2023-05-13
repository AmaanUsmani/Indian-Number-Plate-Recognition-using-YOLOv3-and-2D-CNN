# Automatic Number Plate Recognition (ANPR) System

This project aims to improve the accuracy of ANPR systems by implementing a pipeline that combines YOLOv3 for number plate detection, a trained Convolutional Neural Network (CNN) model for character recognition, and OpenCV's contour detection algorithm for character segmentation. The system's accuracy is compared with EasyOCR and ResNet-50 models for performance evaluation.

## Features

- Utilizes YOLOv3 for efficient and accurate detection of number plates in images.
- Implements OpenCV's contour detection algorithm to accurately segment individual characters from number plates.
- Trains a CNN model on the segmented character dataset to recognize characters with high accuracy.
- Compares the performance of the trained CNN model with EasyOCR and ResNet-50 models using a test set of segmented characters.
- Provides an end-to-end ANPR system for automatic detection and recognition of number plates.

## Installation

1. Clone the repository:
git clone https://github.com/AmaanUsmani/Indian-Number-Plate-Recognition-using-YOLOv3-and-2D-CNN.git

2. Install the required dependencies:
pip install -r requirements.txt

3. Download the pre-trained weights for YOLOv3 and place them in the appropriate directory (model --> weights).
https://drive.google.com/drive/folders/1STio9AvOODRUdLEI0lX2vh1NS0GDeUm5?usp=share_link


