# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 23:27:40 2017

@author: lancel
"""

# import libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator

# Build CNN
cnn_classifier = Sequential()
cnn_classifier.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (64, 64, 3), activation = 'relu'))
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))
cnn_classifier.add(Flatten())
cnn_classifier.add(Dense(output_dim = 128, activation = 'relu'))
cnn_classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# CNN compile
cnn_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# importing images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasets/cnn_dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('datasets/cnn_dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn_classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)