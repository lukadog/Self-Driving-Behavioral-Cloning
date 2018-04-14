# Behaviorial Cloning Project Report

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Files included are a model.py file, drive.py, model.h5 a writeup report and video.mp4.

Overview
---
The object of this project is to train deep learning neural network to drive autonomously in a simulator. 

The goals / steps of this project are the following:
* Use the simulator training mode to collect data of good driving metrics, namely the images and steering angles. This is much more difficult than you think... @@
* Analyze the collected data and apply image preprocessing to make it more suitable for training.
* Design, train and validate a deep learning model that predicts a steering angle from the test image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle is able to remain on the track for one loop.
* Summarize the results with a written report

Steps:
---

### 1. Collecting driving data

I managed to collect the driving data for 5 laps and kept the car in the center of the road... 

The simulator produces 7 colums of data per row. The first three colums are the directory of the images taken by center, left and right mounted cameras respectively. The images are used as input data for training and test the model. The rest of the four colums are steering angle, throttle, brake, and speed for each record. For this project, I used the steering angle as the labeling data.


### 2. Data augmentation

* Because the most turns are left turns, to reduce the bias towards left turn, I have flipped images and taking the opposite sign of the steering measurement.
* Add images taken from the left and right mounted cameras by applying the steering correction angle.

```
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
```


### 3. Process the image

THe images are processed as follows:

* Convert from BGR to HSV space so it's less sensitive to brightness.
* Crop the top 40 pixels and bottom pixels so the focus is more or less on the pavement.
* Resize the image into decent size (200 x 66 x 3) before feeding to the network.

```
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
```

### 4. Network architecture 

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
```

epoch = 30
optimize = adam

### Reduce Overfitting

The way I identify overfitting is by looking at "loss" value and "val_loss" value. If the loss value keeps decreasing but the val_loss value does not change or even increase, that means the model is being overfitted. 

The mitigations:

* Dropout is applied 
* More various data is collected 
* Data is shuffled
* Train/validation/test splits have been used

### Training Process
```
Epoch 00001: val_mean_squared_error improved from inf to 0.05945, saving model to model.h5
Epoch 2/30
26698/26698 [==============================] - 25s 926us/step - loss: 0.0586 - mean_squared_error: 0.0586 - val_loss: 0.0488 - val_mean_squared_error: 0.0488

Epoch 00002: val_mean_squared_error improved from 0.05945 to 0.04880, saving model to model.h5
Epoch 3/30
26698/26698 [==============================] - 25s 928us/step - loss: 0.0483 - mean_squared_error: 0.0483 - val_loss: 0.0412 - val_mean_squared_error: 0.0412

Epoch 00003: val_mean_squared_error improved from 0.04880 to 0.04120, saving model to model.h5
Epoch 4/30
26698/26698 [==============================] - 25s 926us/step - loss: 0.0422 - mean_squared_error: 0.0422 - val_loss: 0.0410 - val_mean_squared_error: 0.0410

Epoch 00004: val_mean_squared_error improved from 0.04120 to 0.04105, saving model to model.h5
Epoch 5/30
26698/26698 [==============================] - 25s 924us/step - loss: 0.0383 - mean_squared_error: 0.0383 - val_loss: 0.0321 - val_mean_squared_error: 0.0321

Epoch 00005: val_mean_squared_error improved from 0.04105 to 0.03206, saving model to model.h5
Epoch 6/30
26698/26698 [==============================] - 25s 923us/step - loss: 0.0350 - mean_squared_error: 0.0350 - val_loss: 0.0335 - val_mean_squared_error: 0.0335

Epoch 00006: val_mean_squared_error did not improve
Epoch 7/30
26698/26698 [==============================] - 25s 919us/step - loss: 0.0317 - mean_squared_error: 0.0317 - val_loss: 0.0297 - val_mean_squared_error: 0.0297

Epoch 00007: val_mean_squared_error improved from 0.03206 to 0.02969, saving model to model.h5
Epoch 8/30
26698/26698 [==============================] - 25s 928us/step - loss: 0.0302 - mean_squared_error: 0.0302 - val_loss: 0.0272 - val_mean_squared_error: 0.0272

Epoch 00008: val_mean_squared_error improved from 0.02969 to 0.02716, saving model to model.h5
Epoch 9/30
26698/26698 [==============================] - 25s 926us/step - loss: 0.0281 - mean_squared_error: 0.0281 - val_loss: 0.0266 - val_mean_squared_error: 0.0266

Epoch 00009: val_mean_squared_error improved from 0.02716 to 0.02657, saving model to model.h5
Epoch 10/30
26698/26698 [==============================] - 25s 919us/step - loss: 0.0267 - mean_squared_error: 0.0267 - val_loss: 0.0242 - val_mean_squared_error: 0.0242

Epoch 00010: val_mean_squared_error improved from 0.02657 to 0.02420, saving model to model.h5
Epoch 11/30
```

