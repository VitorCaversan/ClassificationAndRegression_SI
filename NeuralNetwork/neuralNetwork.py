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
from sklearn.metrics       import mean_squared_error as MSE

class NeuralNet:
    def __init__(self, csvFilePath) -> None:
        # pandas.options.display.max_columns = 50
        # self.fileData = pandas.read_csv(csvFilePath, encoding = 'utf-8')
        self.fileData    = self.manualParsing(csvFilePath)
        self.colNames    = []
        self.input       = []
        self.outputGrav  = []
        self.outputClass = []

        return

    def createModel(self):
        self.colNames = self.fileData[0][3:-1]
        shuffledData  = shuffle(self.fileData[1:])
        # Used with pandas:
        # self.input  = self.fileData[['pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav']] # NN samples
        # self.output = self.fileData['risco'].values # NN labels

        gravRow  = len(self.fileData[0]) - 2
        classRow = len(self.fileData[0]) - 1
        for row in shuffledData:
            self.input.append(row[3:-2])
            self.outputGrav.append(row[gravRow])
            self.outputClass.append(row[classRow] - 1)

        self.input = numpy.array(self.input)
        # self.input = normalize(self.input) # Normalize when values have a great range

        self.outputGrav  = numpy.array(self.outputGrav)
        # An encoding to classify each label with only one bit. Used for multiclass:
        self.outputClass = to_categorical(self.outputClass, num_classes=4, dtype=int)
        self.outputClass = numpy.array(self.outputClass)

        print(self.colNames)
        print(self.input)
        print(self.outputGrav)
        print(self.outputClass)

        self.modelGrav = Sequential([Dense(units=512, input_dim=3, activation='relu'),
                                     Dense(units=256, activation='relu'),
                                     Dense(units=128, activation='relu'),
                                     Dense(units=1, activation='linear')])

        self.modelClass = Sequential([Dense(units=512, input_dim=3, activation='relu'),
                                      Dense(units=256, activation='relu'),
                                      Dense(units=128, activation='relu'),
                                      Dense(units=4, activation='softmax')]) # 'softmax' for multiclass problems

        self.modelGrav.summary()
        self.modelClass.summary()

        return
    
    def runModel(self):
        self.modelGrav.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), # Or rmsprop
                               loss='mean_squared_error', # categorical_crossentropy for multiclass
                               metrics=[tf.keras.metrics.RootMeanSquaredError()])
                                    # Other possible metrics:
                                    # 'accuracy',
                                    # tf.keras.metrics.Accuracy(name="accuracy"),
                                    # tf.keras.metrics.Recall(name="recall")])
                                    # tf.keras.metrics.TruePositives(name="truePositives"),
                                    # tf.keras.metrics.TrueNegatives(name="trueNegatives"),
                                    # tf.keras.metrics.FalsePositives(name="falsePositives"),
                                    # tf.keras.metrics.FalseNegatives(name="falseNegatives")])

        self.modelClass.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), # Or rmsprop
                               loss='categorical_crossentropy', # categorical_crossentropy for multiclass
                               metrics=['accuracy'])

        self.modelGrav.fit(x=self.input,
                           y=self.outputGrav,
                           # steps_per_epoch=3, # completely fucks up the learning process
                           validation_split=0.1,
                           # batch_size=20,
                           epochs=300,
                           shuffle=True,
                           verbose=1)

        self.modelClass.fit(x=self.input,
                           y=self.outputClass,
                           # steps_per_epoch=3, # completely fucks up the learning process
                           validation_split=0.1,
                           # batch_size=20,
                           epochs=300,
                           shuffle=True,
                           verbose=1)

        return

    def predictData(self, rowsQntToPredict: int) -> None:
        predictionsGrav  = self.modelGrav.predict(self.input[:rowsQntToPredict], verbose=2)
        predictionsClass = self.modelClass.predict(self.input[:rowsQntToPredict], verbose=2)
        mse_grav = MSE(self.outputGrav[:rowsQntToPredict], predictionsGrav)
        rmse = mse_grav**(1/2)

        print("Predicted gravity values are: \n", predictionsGrav)
        print("Real values are: \n", self.outputGrav[:rowsQntToPredict])
        print("RMSE = ", rmse)
        print("")

        print("Predicted Classification values are: \n", predictionsClass)
        print("Real values are: \n", self.outputClass[:rowsQntToPredict])
        print("")

        return

    def saveModel(self):
        self.modelGrav.save('models/sinais_vitais_model.h5')
        self.modelClass.save('models/sinais_vitais_modelClass.h5')
        return

    def loadModel(self) -> bool:
        try:
            self.modelGrav  = load_model('models/sinais_vitais_model.h5')
            self.modelClass = load_model('models/sinais_vitais_modelClass.h5')
            gravRow  = len(self.fileData[0]) - 2
            classRow = len(self.fileData[0]) - 1
            for row in self.fileData[1:]:
                self.input.append(row[3:-2])
                self.outputGrav.append(row[gravRow])
                self.outputClass.append(row[classRow] - 1)

            self.input = numpy.array(self.input)
            self.outputGrav  = numpy.array(self.outputGrav)
            # An encoding to classify each label with only one bit. Used for multiclass:
            self.outputClass = to_categorical(self.outputClass, num_classes=4, dtype=int)
            self.outputClass = numpy.array(self.outputClass)
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