# Age-and-Gender-Detection-using-CNN
This project focuses on real-time age and gender detection using Convolutional Neural Networks (CNNs), trained on a dataset of 23,708 labeled images. The key objective was to build a robust model capable of accurately predicting both age and gender, and to implement it in real-time using the YOLO algorithm and a webcam.

Project Highlights:
Dataset: 23,708 images labeled with age and gender.

Data Split: 80% training and 20% testing.

Model Architecture:

4 Convolutional layers with ReLU activations and MaxPooling.

4 Dense (fully connected) layers.

Softmax output layers for classification.

Accuracy:

Gender prediction: 85%

Age prediction: 72%

Training Strategy:

Used batch-wise training with iterations to efficiently handle the large dataset.

Applied data augmentation (rotation, zoom, flip, etc.) to improve model generalization.

Real-Time Detection:

Integrated with YOLO for real-time face detection.

The age and gender predictions are displayed directly from webcam feed.



#age model is quite big to upload.
