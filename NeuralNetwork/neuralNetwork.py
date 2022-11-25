import os
import sys
import pandas # for data manipulation
import numpy # for data manipulation
import tensorflow as tf # the neural network itself
import sklearn
from tensorflow            import keras
from keras.models          import Sequential, load_model
from keras.layers          import Activation, Dense
from keras.utils           import normalize, to_categorical
from sklearn.utils         import shuffle
from sklearn.preprocessing import MinMaxScaler

class NeuralNet:
    def __init__(self, csvFilePath) -> None:
        # pandas.options.display.max_columns = 50
        # self.fileData = pandas.read_csv(csvFilePath, encoding = 'utf-8')
        self.fileData = self.manualParsing(csvFilePath)

        return

    def createModel(self):
        self.fileData = self.fileData[1:]
        shuffledData = shuffle(self.fileData)
        # self.input  = self.fileData[['pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav']] # NN samples
        # self.output = self.fileData['risco'].values # NN labels

        self.input = []
        self.output = []

        lastRow = len(self.fileData[0]) - 1
        for row in self.fileData:
            self.input.append(row[1:-1])
            self.output.append(row[lastRow])

        self.input = numpy.array(self.input)
        self.input = normalize(self.input, axis=1) # Must normalize to work

        self.output = numpy.array(self.output)
        self.output = to_categorical(self.output) # An encoding to classify each label with only one bit

        print(self.input)
        print(self.output)
        # scaler = MinMaxScaler(feature_range=(0,1))
        # self.input = scaler.fit_transform(self.input.reshape(-1,1))

        self.model = Sequential([Dense(units=16, input_shape=(6,), activation='relu'),
                                 Dense(units=32, activation='relu'),
                                 Dense(units=5, activation='softmax')]) # Softmax always recomended for multiclass problems

        self.model.summary()

        return
    
    def runModel(self):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), # Or rmsprop
                           loss='categorical_crossentropy', # Also recomended for multiclass
                           metrics=[tf.keras.metrics.Accuracy(name="accuracy"),
                                    tf.keras.metrics.Recall(name="recall")])
                                    # tf.keras.metrics.TruePositives(name="truePositives"),
                                    # tf.keras.metrics.TrueNegatives(name="trueNegatives"),
                                    # tf.keras.metrics.FalsePositives(name="falsePositives"),
                                    # tf.keras.metrics.FalseNegatives(name="falseNegatives")])

        self.model.fit(x=self.input, y=self.output, steps_per_epoch=3 ,validation_split=0.05, batch_size=20, epochs=100, shuffle=True, verbose=1)

    def manualParsing(self, filePath):
        fileContent = open(filePath, 'r')

        fileData = []
        # Puts it in a matrix
        for line in fileContent:
            fileData.append(line.split(','))

        # Removes '\n' from the last collumn
        rowLength = len(fileData[0]) - 1
        for row in range(len(fileData)):
            number = fileData[row][rowLength]
            fileData[row][rowLength] = number[:-1]

        # Casts inputs to float
        for colIterator in range(len(fileData[0])-1):
            for rowIterator in range(len(fileData)):
                try:
                    fileData[rowIterator][colIterator] = float(fileData[rowIterator][colIterator])
                except:
                    pass
        # Casts outputs to int
        for rowIterator in range(len(fileData)):
            try:
                fileData[rowIterator][rowLength] = int(fileData[rowIterator][rowLength])
            except:
                pass

        return fileData