# Traffic Sign Recognition with Deep Learning

This project involves building a convolutional neural network (CNN) to classify images of traffic signs. Using the GTSRB dataset, the model is trained to recognize and categorize traffic signs into their respective classes.

## Features

- **Deep Learning**: Implements a CNN architecture using TensorFlow and Keras.
- **Traffic Sign Classification**: Trains the model to recognize and categorize over 40 types of traffic signs.
- **Model Evaluation**: Measures model performance using accuracy and loss metrics on training and testing datasets.

## Project Files

- `traffic.py`: The main script for data processing, model training, and evaluation.
- `gtsrb/`: Folder containing the GTSRB traffic sign dataset.
- `model.h5`: Saved model file after training.
- `README.md`: Documentation for the project.

## How It Works

1. **Dataset**:
   - The GTSRB dataset contains labeled images of various traffic signs.
   - Images are preprocessed to ensure consistency in size and format.

2. **Model Architecture**:
   - A convolutional neural network is implemented with multiple convolutional, pooling, and dense layers.
   - The model is designed to learn spatial hierarchies from image data.

3. **Training**:
   - The model is trained on labeled images for 10 epochs (or more, as configured).
   - Loss and accuracy are monitored to prevent overfitting.

4. **Evaluation**:
   - The model's accuracy is calculated on a separate test set.
   - Results highlight the model's ability to generalize and predict unseen data.

## Dataset Download

Due to size constraints, the dataset is not included in this repository. You can download the GTSRB dataset using the following link:

[Download GTSRB Dataset](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip)

Once downloaded, extract the dataset into the project directory.
