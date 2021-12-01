import pandas as pd
from curses import KEY_SAVE
from unittest.mock import NonCallableMagicMock
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
import tensorflow
from tensorflow.keras import Model
import os
import numpy as np
os.environ["RF_CPP_MIN_LEVEL"] = "8"


class CNNBlock(layers.Layer):
    """Creates a CNN block consisting of convolutional layer followed by optional max pooling layer.
    :param layers: inherited from Tensorflow base class
    :type layers: module
    """

    def __init__(self, n_filters=16, kernel_size=2, pool_size=2, padding='same', batch_norm=False):
        """
        :param n_filters: number of filters to output, defaults to 16
        :type n_filters: int
        :param kernel_size: dimensions of the convolution filter, defaults to 3
        :type kernel_size: int or shape, optional
        :param pool_size: size of the pool filer
        :type pool_size: int or shape
        :param padding: whether to pad inputs to layer
        :type padding: string, ['same', 'valid']
        :param batch_norm: batch normalization on layer outputs, defaults to False
        :type batch_nrom: bool

        """
        super(CNNBlock, self).__init__()
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.padding = padding
        self.batch_norm = batch_norm

        # CNN layer
        self.conv = layers.Conv2D(
            filters=self.n_filters, kernel_size=self.kernel_size,
            padding=self.padding, activation='relu')

        # Maxpool layer
        self.pool = layers.MaxPooling2D(
            pool_size=self.pool_size, padding=self.padding, strides=2)
        self.norm = tensorflow.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        # Convolution
        x = self.conv(x, training=training)
        # Max pooling
        #x = self.pool(x, training=training)
        if self.batch_norm:
            x = self.norm(x)
        return x


class DNNBlock(layers.Layer):
    """Creates a deconvolutional block to upsample encoded CNN latent space.
    :param layers: inherited from Tensorflow base class
    :type layers: module
    """

    def __init__(self, n_filters=16, kernel_size=2, pool_size=2, padding='same', activation='relu', batch_norm=False):
        """
         :param n_filters: number of filters to output, defaults to 16
        :type n_filters: int
        :param kernel_size: dimensions of the convolution filter, defaults to 3
        :type kernel_size: int or shape, optional
        :param pool_size: size of the pool filer
        :type pool_size: int or shape
        :param padding: whether to pad inputs to layer
        :type padding: string, ['same', 'valid']
        :param activation: activation fn to apply, defaults to 'relu'
        :type batch_norm: string
        :param batch_norm: batch normalization on layer outputs, defaults to False
        :type batch_norm: bool        
        """
        super(DNNBlock, self).__init__()
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.padding = padding
        self.activation = activation
        self.batch_norm = batch_norm

        # Cnn layers
        self.conv = layers.Conv2D(filters=self.n_filters, kernel_size=self.kernel_size,
                                  padding=self.padding, activation=self.activation)
        self.upsample = layers.UpSampling2D(size=self.pool_size)
        self.norm = tensorflow.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        #x = self.upsample(x, training=training)

        if self.batch_norm:
            x = self.norm(x)
        return x


class CAE(Model):
    """Convolutional Autoencoder (CAE)
    :param Model: Inherited from Tensorflow base class]
    :type Model: Module
    """

    def __init__(self):
        super(CAE, self).__init__()
        # Encoding layer
        self.enc1 = CNNBlock(n_filters=22**2, kernel_size=2,
                             pool_size=2, padding='same')
        self.enc2 = CNNBlock(n_filters=200, kernel_size=2,
                             pool_size=2, padding='same')
        self.enc3 = CNNBlock(n_filters=100, kernel_size=2,
                             pool_size=2, padding='same')
        self.enc4 = CNNBlock(n_filters=50, kernel_size=2,
                             pool_size=2, padding='same')
        self.enc5 = CNNBlock(n_filters=10, kernel_size=2,
                             pool_size=2, padding='same')

        # Decoding layers
        self.dec1 = DNNBlock(n_filters=10, kernel_size=2,
                             pool_size=2,  padding='same')
        self.dec2 = DNNBlock(n_filters=50, kernel_size=2,
                             pool_size=2,  padding='same')
        self.dec3 = DNNBlock(n_filters=100, kernel_size=2,
                             pool_size=2, padding='same')
        self.dec4 = DNNBlock(n_filters=200, kernel_size=2,
                             pool_size=2, padding='same')
        self.dec5 = DNNBlock(n_filters=22**2, kernel_size=2,
                             pool_size=2, padding='same')
        self.out = layers.Conv2D(
            1, 3, activation='sigmoid', padding='same', name='outputs')

    def call(self, x, training=False):
        x = self.enc1(x, training=training)
        x = self.enc2(x, training=training)
        x = self.enc3(x, training=training)
        #x = self.enc4(x, training=training)
        #x = self.enc5(x, training=training)

        # Latent representation

        #x = self.dec1(x, training=training)
        #x = self.dec2(x, training=training)
        x = self.dec3(x, training=training)
        x = self.dec4(x, training=training)
        x = self.dec5(x, training=training)
        x = self.out(x, training=training)
        return x

    def get_model(self, input_shape=(28, 28, 1)):
        """Return the model - work in progress

        :param input_shape: input dimensions, defaults to (28, 28, 1)
        :type input_shape: tuple, optional
        :return: full autoencoder model and summary statement
        :rtype: model object
        """
        # AutoEncoder model
        x = tensorflow.keras.Input(shape=input_shape)
        ae = tensorflow.keras.Model(x, self.call(x), name="full")
        print(ae.summary())
        return ae
