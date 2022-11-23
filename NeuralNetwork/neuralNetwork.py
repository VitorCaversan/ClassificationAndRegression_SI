import pandas # for data manipulation
import numpy # for data manipulation
import tensorflow as tf # the neural network itself
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
import os
import sys

class NeuralNet:
    def __init__(self, csvFilePath) -> None:
        pandas.options.display.max_columns = 50

        self.fileData = pandas.read_csv(csvFilePath, encoding = 'utf-8')

        return

    def createModel(self):
        self.input  = self.fileData[['pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav']]
        self.output = self.fileData['risco'].values

        self.model = Sequential([Dense(units=16, input_shape=(1,), activation='relu'),
                                 Dense(units=32, activation='relu'),
                                 Dense(units=4, activation='softmax')])