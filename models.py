"""
A collection of models we'll use to attempt to classify videos.
"""

import keras
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization, Input, concatenate, \
    GlobalAveragePooling2D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, AveragePooling2D, MaxPooling2D)
from collections import deque
from keras.backend import tensorflow_backend
import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.set_random_seed(0)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import sys


class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
        """
        `model` = one of:
            lstm
            crnn
            mlp
            conv_3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        # if self.nb_classes >= 10:
        #     metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'crnn':
            print("Loading CRNN model.")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.crnn()
        elif model == 'crnn1':
            print("Loading CRNN1 model.")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.crnn1()
        elif model == 'crnn2':
            print("Loading CRNN2 model.")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.crnn2()
        elif model == 'crnn3':
            print("Loading CRNN3 model.")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.crnn3()
        elif model == 'crnn4':
            print("Loading CRNN4 model.")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.crnn4()
        elif model == 'bicrnn':
            print("Loading BiCRNN model.")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.bicrnn()
        elif model == 'inception':
            print('Loading Inception model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception()
        elif model == 'inception1':
            print('Loading Inception1 model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception1()
        elif model == 'inception2':
            print('Loading Inception2 model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception2()
        elif model == 'inception3':
            print('Loading Inception3 model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception3()
        elif model == 'inception4':
            print('Loading Inception4 model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception4()
        elif model == 'inception3_simple':
            print('Loading Inception3_simple model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception3_simple()
        elif model == 'inception_gru':
            print('Loading Inception_GRU model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception_gru()
        elif model == 'inception_gru3':
            print('Loading Inception_GRU3 model.')
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.inception_gru3()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = features_length * seq_length
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.conv_3d()
        elif model == 'div_crnn':
            print("Loading DIV_CRNN")
            self.input_shape = (seq_length, 16, 16, 1)
            self.model = self.div_crnn()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-4)  # aggressively small learning rate
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=True, input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def crnn(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def crnn1(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu')))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def crnn2(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        model = Sequential()
        model.add(TimeDistributed(Conv2D(64, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(64, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def crnn3(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu')))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def crnn4(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        model = Sequential()
        model.add(TimeDistributed(Conv2D(64, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def bicrnn(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal",
                                         activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def inception(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(Conv2D(64, (5, 5), padding='same', activation='relu'))(tower_2)

        tower_3 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input_img)
        tower_3 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_3)
        tower_3 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(tower_3)

        x1 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        tower_4 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(x1)
        tower_4 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_4)
        tower_4 = keras.layers.TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(tower_4)

        tower_5 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(x1)
        tower_5 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_5)
        tower_5 = keras.layers.TimeDistributed(Conv2D(64, (5, 5), padding='same', activation='relu'))(tower_5)

        tower_6 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(x1)
        tower_6 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_6)
        tower_6 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(tower_6)

        x2 = keras.layers.concatenate([tower_4, tower_5, tower_6], axis=3)

        # x = keras.layers.TimeDistributed(Flatten())(x2)
        x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x2)
        x = LSTM(256, return_sequences=True)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Dropout(0.5)(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def inception1(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_2)

        tower_3 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input_img)
        tower_3 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_3)
        tower_3 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_3)

        x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
        # x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x)
        x = keras.layers.TimeDistributed(Flatten())(x)
        x = LSTM(256, return_sequences=True)(x)
        x = Flatten()(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def inception2(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_2)

        # tower_3 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input_img)
        # tower_3 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_3)
        # tower_3 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(tower_3)

        x1 = keras.layers.concatenate([tower_1, tower_2], axis=2)

        tower_4 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(x1)
        tower_4 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_4)
        tower_4 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_4)

        tower_5 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(x1)
        tower_5 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_5)
        tower_5 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_5)

        # tower_6 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(x1)
        # tower_6 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_6)
        # tower_6 = keras.layers.TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(tower_6)

        x2 = keras.layers.concatenate([tower_4, tower_5], axis=2)

        # x = keras.layers.TimeDistributed(Flatten())(x2)
        x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x2)
        x = LSTM(256, return_sequences=True)(x)
        x = Flatten()(x)
        # x = Dense(512)(x)
        x = Dropout(0.5)(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def inception3(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_2)

        tower_3 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input_img)
        tower_3 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_3)
        tower_3 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_3)

        x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x)
        # x = keras.layers.TimeDistributed(Flatten())(x)
        x = LSTM(256, return_sequences=True)(x)
        x = Flatten()(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def inception3_simple(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_2)

        tower_3 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input_img)
        tower_3 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_3)
        tower_3 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_3)

        x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x)
        # x = keras.layers.TimeDistributed(Flatten())(x)
        x = SimpleRNN(256, return_sequences=True)(x)
        x = Flatten()(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def inception4(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_2)

        tower_3 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input_img)
        tower_3 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_3)
        tower_3 = keras.layers.TimeDistributed(Conv2D(32, (1, 1), padding='same', activation='relu'))(tower_3)

        x1 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        tower_4 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(x1)
        tower_4 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_4)
        tower_4 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_4)

        tower_5 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(x1)
        tower_5 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_5)
        tower_5 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_5)

        tower_6 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(x1)
        tower_6 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_6)
        tower_6 = keras.layers.TimeDistributed(Conv2D(32, (1, 1), padding='same', activation='relu'))(tower_6)

        x2 = keras.layers.concatenate([tower_4, tower_5, tower_6], axis=3)

        # x = keras.layers.TimeDistributed(Flatten())(x2)
        x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x2)
        x = LSTM(256, return_sequences=True)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Dropout(0.5)(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def inception_gpu(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_2)

        x1 = keras.layers.concatenate([tower_1, tower_2], axis=2)

        tower_4 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(x1)
        tower_4 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_4)
        tower_4 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_4)

        tower_5 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(x1)
        tower_5 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_5)
        tower_5 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_5)

        x2 = keras.layers.concatenate([tower_4, tower_5], axis=2)

        # x = keras.layers.TimeDistributed(Flatten())(x2)
        x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x2)
        x = GRU(256, return_sequences=True)(x)
        x = Flatten()(x)
        # x = Dense(512)(x)
        x = Dropout(0.5)(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def inception_gru3(self):
        input_img = Input(shape=self.input_shape)

        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_1 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_1)
        tower_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_1)

        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        tower_2 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_2)
        tower_2 = keras.layers.TimeDistributed(
            Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_2)

        tower_3 = keras.layers.TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input_img)
        tower_3 = keras.layers.TimeDistributed(BatchNormalization(axis=3, scale=False))(tower_3)
        tower_3 = keras.layers.TimeDistributed(
            Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'))(tower_3)

        x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        x = keras.layers.TimeDistributed(GlobalAveragePooling2D())(x)
        # x = keras.layers.TimeDistributed(Flatten())(x)
        x = GRU(256, return_sequences=True)(x)
        x = Flatten()(x)

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=out)

        return model

    def mlp(self):
        """Build a simple MLP."""
        # Model.
        model = Sequential()
        model.add(Dense(512, input_dim=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            32, (7, 7, 7), activation='relu', input_shape=self.input_shape
        ))
        model.add(Conv3D(64, (3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def div_crnn(self):
        input_img = Input(shape=self.input_shape)

        layer_1 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(input_img)
        layer_2 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))(layer_1)
        layer_3 = keras.layers.TimeDistributed(Flatten())(layer_2)
        layer_4 = LSTM(256, return_sequences=True)(layer_3)
        x = Flatten()(layer_4)
        x1 = Dense(256)(x)
        x2 = Dropout(0.5)(x)
        out1 = Dense(self.nb_classes, activation='softmax')(x2)

        y = Dense(256)(x)
        y = Dropout(0.5)(y)
        out2 = Dense(2, activation='softmax')(y)

        model = Model(inputs=input_img, outputs=[out1, out2])

        return model
