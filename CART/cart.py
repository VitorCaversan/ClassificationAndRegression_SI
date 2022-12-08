import pandas  # for data manipulation
import numpy  # for data manipulation

# for splitting the data into train and test samples
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, mean_absolute_error, mean_squared_log_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error, max_error  # for model evaluation metrics
from sklearn import tree  # for decision tree models

import plotly.express as express  # for data visualization
import plotly.graph_objects as graph  # for data visualization
import graphviz  # for plotting decision tree graphs

import os
import sys

class Cart:
    def __init__(self, csvFilePath: str) -> None:
        # To display more collumns
        pandas.options.display.max_columns = 50

        self.fileData = pandas.read_csv(csvFilePath, encoding='utf-8')

        print(self.fileData)

    def doStuff(self) -> None:
        # classificador
        print('=====================================')
        print('CLASSIFICADOR')
        print('=====================================')
        input = self.fileData[['qPA', 'pulso', 'resp']]
        output = self.fileData['risco'].values

        input_train, input_test, output_train, output_test, clf = self.fitting(
            input, output, "entropy", 7)

        # regressor
        print('=====================================')
        print('REGRESSOR')
        print('=====================================')
        outputRegressor = self.fileData['grav'].values
        self.regressor(input, outputRegressor, "squared_error", 7)

    def regressor(self, input, output, criterio, mdepth):

        input_train, input_test, output_train, output_test = train_test_split(
            input, output, test_size=0.3)

        # Fit the model
        model = tree.DecisionTreeRegressor(
            criterion=criterio, max_depth=mdepth)
        clf = model.fit(input_train, output_train)

        # Predict class labels on training data
        pred_labels_tr = model.predict(input_train)
        # Predict class labels on a test data
        pred_labels_te = model.predict(input_test)

        # Tree summary and model evaluation metrics
        print('*************** Tree Summary ***************')
        # print('Classes: ', clf.classes_)
        print('Tree Depth: ', clf.tree_.max_depth)
        print('No. of leaves: ', clf.tree_.n_leaves)
        print('No. of features: ', clf.n_features_in_)
        print('--------------------------------------------------------')
        print("")

        print('*************** Evaluation on Training Data ***************')
        score_tr = model.score(input_train, output_train)
        print('Accuracy Score: ', score_tr)
        # Look at classification report to evaluate the model
        print('Mean Absolute Error: ',
              mean_absolute_error(output_train, pred_labels_tr))
        print('Mean Squared Error: ', mean_squared_error(
            output_train, pred_labels_tr))
        print('Mean Absolute Percentage Error: ',
              mean_absolute_percentage_error(output_train, pred_labels_tr))
        print('Median Absolute Error: ',
              median_absolute_error(output_train, pred_labels_tr))
        print('Max Error: ', max_error(output_train, pred_labels_tr))
        print('--------------------------------------------------------')

        print('*************** Evaluation on Test Data ***************')
        score_new = model.score(input_test, output_test)
        print('Accuracy Score: ', score_new)
        # Look at classification report to evaluate the model
        print('Mean Absolute Error: ', mean_absolute_error(
            output_test, pred_labels_te))
        print('Mean Squared Error: ', mean_squared_error(
            output_test, pred_labels_te))
        print('Mean Absolute Percentage Error: ',
              mean_absolute_percentage_error(output_test, pred_labels_te))
        print('Median Absolute Error: ',
              median_absolute_error(output_test, pred_labels_te))
        print('Max Error: ', max_error(output_test, pred_labels_te))
        print('--------------------------------------------------------')
        print("")

        # TESTE
        fileData2 = self.manualParsing('../tar2_sinais_vitais_teste_com_label.txt')

        x = []
        y = []
        gravCol  = len(fileData2[0]) - 2
        for row in fileData2[1:]:
            x.append(row[3:-2])
            y.append(row[gravCol])

        pred_labels_new = model.predict(x)

        print('*************** Evaluation on New Data ***************')
        score_new = model.score(x, y)
        print('Accuracy Score: ', score_new)
        # Look at classification report to evaluate the model
        print('Mean Absolute Error: ', mean_absolute_error(y, pred_labels_new))
        print('Mean Squared Logarithmic Error: ',
              mean_squared_log_error(y, pred_labels_new))
        print('Mean Squared Error: ', mean_squared_error(y, pred_labels_new))
        print('Mean Absolute Percentage Error: ',
              mean_absolute_percentage_error(y, pred_labels_new))
        print('Median Absolute Error: ',
              median_absolute_error(y, pred_labels_new))
        print('Max Error: ', max_error(y, pred_labels_new))
        print('--------------------------------------------------------')
        print("")

        ######################
    def fitting(self, input, output, criterio, mdepth):

        # Create training and testing samples
        input_train, input_test, output_train, output_test = train_test_split(
            input, output, test_size=0.3)

        # Fit the model
        model = tree.DecisionTreeClassifier(criterion=criterio,
                                            max_depth=mdepth)
        clf = model.fit(input_train, output_train)

        # Predict class labels on training data
        pred_labels_tr = model.predict(input_train)
        # Predict class labels on a test data
        pred_labels_te = model.predict(input_test)

        # Tree summary and model evaluation metrics
        print('*************** Tree Summary ***************')
        print('Classes: ', clf.classes_)
        print('Tree Depth: ', clf.tree_.max_depth)
        print('No. of leaves: ', clf.tree_.n_leaves)
        print('No. of features: ', clf.n_features_in_)
        print('--------------------------------------------------------')
        print("")

        print('*************** Evaluation on Test Data ***************')
        score_te = model.score(input_train, output_train)
        print('Accuracy Score: ', score_te)
        # Look at classification report to evaluate the model
        print(classification_report(output_train, pred_labels_tr))
        print("f-measure (weighted):", f1_score(output_train,
              pred_labels_tr, average="weighted"))
        print('--------------------------------------------------------')
        print("")

        print('*************** Evaluation on Training Data ***************')
        score_tr = model.score(input_test, output_test)
        print('Accuracy Score: ', score_tr)
        # Look at classification report to evaluate the model
        print(classification_report(output_test, pred_labels_te))
        print("f-measure (weighted):", f1_score(output_test,
              pred_labels_te, average="weighted"))
        print('--------------------------------------------------------')
        print("")

        fileData2 = self.manualParsing('../tar2_sinais_vitais_teste_com_label.txt')
        x = []
        y = []

        classCol = len(fileData2[0]) - 1
        for row in fileData2[1:]:
            x.append(row[3:-2])
            y.append(row[classCol])

        pred_labels_new = model.predict(x)

        print('*************** Evaluation on New Data ***************')
        score_new = model.score(x, y)
        print('Accuracy Score: ', score_new)
        # Look at classification report to evaluate the model
        print(classification_report(y, pred_labels_new))
        print("f-measure (weighted):", f1_score(y,
              pred_labels_new, average="weighted"))

        print('--------------------------------------------------------')
        # Use graphviz to plot the tree
        # dot_data = tree.export_graphviz(clf, out_file=None,
        #                            feature_names=input.columns,
        #                            class_names=[str(list(clf.classes_)[0]), str(list(clf.classes_)[1])],
        #                            filled=True,
        #                            rounded=True,
        #                            #rotate=True,
        #                        )
        # graph = graphviz.Source(dot_data)

        # Return relevant data for chart plotting
        return input_train, input_test, output_train, output_test, clf  # , graph

    def Plot_3D(self, X, X_test, y_test, clf, x1, x2, mesh_size, margin):

        # Specify a size of the mesh to be used
        mesh_size = mesh_size
        margin = margin

        # Create a mesh grid on which we will run our model
        x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min(
        ) - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
        y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min(
        ) - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
        xrange = numpy.arange(x_min, x_max, mesh_size)
        yrange = numpy.arange(y_min, y_max, mesh_size)
        xx, yy = numpy.meshgrid(xrange, yrange)

        # Calculate predictions on grid
        Z = clf.predict_proba(numpy.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Create a 3D scatter plot with predictions
        fig = express.scatter_3d(x=X_test[x1], y=X_test[x2], z=y_test,
                                 opacity=0.8, color_discrete_sequence=['black'])

        # Set figure title and colors
        fig.update_layout(  # title_text="Scatter 3D Plot with CART Prediction Surface",
            paper_bgcolor='white',
            scene=dict(xaxis=dict(title=x1,
                                  backgroundcolor='white',
                                  color='black',
                                  gridcolor='#f0f0f0'),
                       yaxis=dict(title=x2,
                                  backgroundcolor='white',
                                  color='black',
                                  gridcolor='#f0f0f0'
                                  ),
                       zaxis=dict(title='Probability of Rain Tomorrow',
                                  backgroundcolor='lightgrey',
                                  color='black',
                                  gridcolor='#f0f0f0',
                                  )))

        # Update marker size
        fig.update_traces(marker=dict(size=1))

        # Add prediction plane
        fig.add_traces(graph.Surface(x=xrange, y=yrange, z=Z, name='CART Prediction',
                                     colorscale='Jet',
                                     reversescale=True,
                                     showscale=False,
                                     contours={"z": {"show": True, "start": 0.5, "end": 0.9, "size": 0.5}}))
        fig.show()
        return fig

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
