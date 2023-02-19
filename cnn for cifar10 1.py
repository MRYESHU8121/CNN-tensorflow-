# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:57:45 2023

@author: YESHU
"""

import tensorflow as tf
from tensorflow.keras import layers

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train.shape

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Define the model architecture
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))


# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Train the model for 10 epochs
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))




#In this model, we first load the CIFAR-10 dataset and normalize the pixel values to be between 0 and 1.
# We then define the architecture of our model with three convolutional layers
# and two max pooling layers, followed by two fully connected layers.
# The output layer has 10 units for the 10 classes in the CIFAR-10 dataset. 
#We then compile the model with the categorical cross-entropy loss and the Adam optimizer, 
#and train the model for 10 epochs.