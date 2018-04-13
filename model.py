import csv
import cv2
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle
import tensorflow as tf


def load_cvs_data(csv_file, correction = 0.1):
    """
    this function returns X_train and y_train data loaded from csv file
    :param csv_file: csv file name and directory
    :param correction: steering angle correction for images taken by the left and right cameras
    :return: X_train (images) and y_train (steering angle)
    """

    # read lines from csv file
    lines = []
    with open(csv_file) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    # define empty arrays to store images and steering angles
    images = []
    measurements = []

    for line in lines:
        measurement = float(line[3])

        # Load center image and steering angle
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        originalImage = cv2.imread(current_path)
        # Apply image preprocessing
        image = preproccess_image(originalImage)
        images.append(image)
        measurements.append(measurement)
        # Add flipped image and steering angle
        images.append(cv2.flip(image, 1))
        measurements.append(-measurement)

        # Load left image and steering angle
        source_path = line[1]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        originalImage = cv2.imread(current_path)
        # Apply image preprocessing
        image = preproccess_image(originalImage)
        images.append(image)
        measurements.append(measurement + correction)
        # Add flipped image and steering angle
        images.append(cv2.flip(image, 1))
        measurements.append(-(measurement + correction))

        # Load right image and steering angle
        source_path = line[2]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        originalImage = cv2.imread(current_path)
        # Apply image preprocessing
        image = preproccess_image(originalImage)
        images.append(image)
        measurements.append(measurement - correction)
        # Add flipped image and steering angle
        images.append(cv2.flip(image, 1))
        measurements.append(-(measurement - correction))

    # convert arrays to numpy arrays that tensor flow accept 
    X_train = np.array(images)
    y_train = np.array(measurements)

    # Uncomment this part to verify the shape of loaded data set
    # print(X_train.shape)
    # print(y_train.shape)

    return X_train, y_train


def preproccess_image(image):
    """
    this function applies some basic image processings to the original images
    :param image: original image
    :return: processed image
    """

    # covert BGR image to RGB space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image.shape)

    # covert RGB image to HSV space 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # print(image.shape)

    # crop off the top 40 pixels and bottom 20 pixels 
    image_crop = image[40:-20, :]
    # print(image_crop.shape)
    
    # resize the image into 200 by 66 
    image_resize = cv2.resize(image_crop, (200, 66), interpolation = cv2.INTER_AREA)
    # print(image_resize.shape)

    return image_resize


def build_model():
    """
    build the model
    """
    model = Sequential() 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (66, 200, 3)))
    model.add(Conv2D(24, 5, 5, activation = 'elu', subsample = (2, 2)))
    model.add(Conv2D(36, 5, 5, activation = 'elu', subsample = (2, 2)))
    model.add(Conv2D(48, 5, 5, activation = 'elu', subsample = (2, 2)))
    model.add(Conv2D(64, 3, 3, activation = 'elu'))
    model.add(Conv2D(64, 3, 3, activation = 'elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))
    model.summary()

    return model


if __name__ == '__main__':

    print("Loading data...")
    # load data
    X_train, y_train = load_cvs_data('driving_log.csv')
    # shuffle the loaded data
    X_train, y_train = shuffle(X_train, y_train)
    # define the model architecture
    model = build_model()
    # compile the model
    model.compile('adam', 'mean_squared_error', ['mean_squared_error'])

    checkpoint = ModelCheckpoint("model.h5", monitor = 'val_mean_squared_error', verbose = 1,
                                  save_best_only = True, mode = 'min')
    
    early_stop = EarlyStopping(monitor = 'val_mean_squared_error', min_delta = 0.0001, patience = 4,
                                verbose = 1, mode = 'min')
    # train the model
    model.fit(X_train, y_train, batch_size = 128, nb_epoch = 30, verbose = 1,
                      callbacks = [checkpoint, early_stop], validation_split = 0.15, shuffle = True)

    print("Saving model...")

    with open("model.json", 'w') as outfile:
        outfile.write(model.to_json())

    print("Finished.")
