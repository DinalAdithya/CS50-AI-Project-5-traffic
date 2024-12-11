import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save('E:/CS50 AI/Project 5/traffic/model.keras')
        print(f"Model saved to {filename}.")


def load_data(data_dir):

    images = []
    labels = []

    for i in range(NUM_CATEGORIES):

        category_path = os.path.join(data_dir, str(i))  # load image file with in those folders
        files = os.listdir(category_path)

        if not os.path.exists(category_path) or not os.listdir(category_path):
            continue

        for file in files:
            # print(f"Processing category {i},file {file}")
            file_path = os.path.join(category_path, file)

            if file.lower().endswith('.ppm'):  # to ensure it's a valid img extenction
                image = cv2.imread(file_path)  # Load the image

                if image is not None:
                    resized_img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # resize each image

                    images.append(resized_img)  # append image
                    labels.append(i)  # append current category

                else:
                    print(f"Faild to load the image: {file_path}")

    return images, labels


def get_model():


    model = Sequential()

    # Convolution Layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    # Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening the Data
    model.add(Flatten())

    # Adding Dense Layer
    model.add(Dense(128, activation='relu'))

    # Output layer for classification (softmax for multiclass classification)
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))

    # Compiling the Model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model




if __name__ == "__main__":
    main()
