import pandas # for data manipulation
import numpy # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models

import plotly.express as express # for data visualization
import plotly.graph_objects as graph # for data visualization
import graphviz # for plotting decision tree graphs

class Cart:
    def __init__(self) -> None:
        # To display more collumns
        pandas.options.display.max_columns = 50

        self.fileData = pandas.read_csv('../tar2_sinais_vitais_treino_com_label.txt', encoding = 'utf-8')

        self.fileData