import pandas # for data manipulation
import numpy # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models

import plotly.express as express # for data visualization
import plotly.graph_objects as graph # for data visualization
import graphviz # for plotting decision tree graphs

import os
import sys

class Cart:
    def __init__(self) -> None:
        # To display more collumns
        pandas.options.display.max_columns = 50

        self.fileData = pandas.read_csv('../tar2_sinais_vitais_treino_com_label.txt', encoding = 'utf-8')

        print(self.fileData)

    def doStuff(self) -> None:
        input  = self.fileData[['pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav']]
        output = self.fileData['risco'].values

        input_train, input_test, output_train, output_test, clf, graph = self.fitting(input, output, "gini", "best", 3, None, 1000)

    def fitting(self, input, output, criterio, splitt, mdepth, clweight, minleaf):

        # Create training and testing samples
        input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=0)

        # Fit the model
        model = tree.DecisionTreeClassifier(criterion=criterio, 
                                            splitter=splitt, 
                                            max_depth=mdepth,
                                            class_weight=clweight,
                                            min_samples_leaf=minleaf, 
                                            random_state=0)
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
        score_te = model.score(input_test, output_test)
        print('Accuracy Score: ', score_te)
        # Look at classification report to evaluate the model
        print(classification_report(output_test, pred_labels_te))
        print('--------------------------------------------------------')
        print("")
        
        print('*************** Evaluation on Training Data ***************')
        score_tr = model.score(input_train, output_train)
        print('Accuracy Score: ', score_tr)
        # Look at classification report to evaluate the model
        print(classification_report(output_train, pred_labels_tr))
        print('--------------------------------------------------------')
        
        # Use graphviz to plot the tree
        dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=input.columns, 
                                    class_names=[str(list(clf.classes_)[0]), str(list(clf.classes_)[1])],
                                    filled=True, 
                                    rounded=True, 
                                    #rotate=True,
                                ) 
        graph = graphviz.Source(dot_data)
        
        # Return relevant data for chart plotting
        return input_train, input_test, output_train, output_test, clf, graph

    def Plot_3D(self, X, X_test, y_test, clf, x1, x2, mesh_size, margin):
                
        # Specify a size of the mesh to be used
        mesh_size=mesh_size
        margin=margin

        # Create a mesh grid on which we will run our model
        x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
        y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
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
        fig.update_layout(#title_text="Scatter 3D Plot with CART Prediction Surface",
                        paper_bgcolor = 'white',
                        scene = dict(xaxis=dict(title=x1,
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
                                contours = {"z": {"show": True, "start": 0.5, "end": 0.9, "size": 0.5}}))
        fig.show()
        return fig

    def manualParsing(self, filePath) -> None:
        fileContent = open(filePath, 'r')

        # Puts it in a matrix
        for line in fileContent:
            self.fileData.append(line.split(','))

        # Removes '\n' from the last collumn
        rowLength = len(self.fileData[0]) - 1
        for row in range(len(self.fileData)):
            number = self.fileData[row][rowLength]
            self.fileData[row][rowLength] = number[:-1]

        # Casts everything to int
        for colIterator in range(len(self.fileData[0])):
            for rowIterator in range(len(self.fileData)):
                try:
                    self.fileData[rowIterator][colIterator] = float(self.fileData[rowIterator][colIterator])
                except:
                    pass