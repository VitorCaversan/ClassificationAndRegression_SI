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
        self.colNames = []
        self.input    = []
        self.output   = []

        return

    def createModel(self):
        self.colNames = self.fileData[0][3:-1]
        shuffledData  = shuffle(self.fileData[1:])
        # Used with pandas:
        # self.input  = self.fileData[['pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav']] # NN samples
        # self.output = self.fileData['risco'].values # NN labels

        gravRow = len(self.fileData[0]) - 2
        for row in shuffledData:
            self.input.append(row[3:-2])
            self.output.append(row[gravRow])

        self.input = numpy.array(self.input)
        # self.input = normalize(self.input) # Normalize when values have a great range

        self.output = numpy.array(self.output)
        # An encoding to classify each label with only one bit. Used for multiclass:
        # self.output = to_categorical(self.output, num_classes=4, dtype=int)

        print(self.colNames)
        print(self.input)
        print(self.output)

        self.model = Sequential([Dense(units=128, input_dim=3, activation='relu'),
                                 Dense(units=64, activation='relu'),
                                 Dense(units=32, activation='relu'),
                                 Dense(units=1, activation='linear')]) # 'softmax' for multiclass problems

        self.model.summary()

        return
    
    def runModel(self):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), # Or rmsprop
                           loss='mean_squared_error', # categorical_crossentropy for multiclass
                           metrics=['mae'])
                                    # Other possible metrics:
                                    # 'accuracy',
                                    # tf.keras.metrics.Accuracy(name="accuracy"),
                                    # tf.keras.metrics.Recall(name="recall")])
                                    # tf.keras.metrics.TruePositives(name="truePositives"),
                                    # tf.keras.metrics.TrueNegatives(name="trueNegatives"),
                                    # tf.keras.metrics.FalsePositives(name="falsePositives"),
                                    # tf.keras.metrics.FalseNegatives(name="falseNegatives")])

        self.model.fit(x=self.input,
                       y=self.output,
                       # steps_per_epoch=3, # completely fucks up the learning process
                       validation_split=0.1,
                       # batch_size=20,
                       epochs=300,
                       shuffle=True,
                       verbose=1)

        return

    def predictData(self, rowsQntToPredict: int) -> None:
        predictions = self.model.predict(self.input[:rowsQntToPredict])

        print("Predicted values are: \n", predictions)
        print("Real values are: \n", self.output[:rowsQntToPredict])
        print("")

        return

    def saveModel(self):
        self.model.save('models/sinais_vitais_model.h5')
        return

    def loadModel(self) -> bool:
        try:
            self.model  = load_model('models/sinais_vitais_model.h5')
            gravRow = len(self.fileData[0]) - 2
            for row in self.fileData[1:]:
                self.input.append(row[3:-2])
                self.output.append(row[gravRow])
            return True
        except:
            print("file was not found \n")
            return False

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