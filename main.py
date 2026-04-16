from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class SequentialModel(object):
    '''A class that manages a certain sequence of layers in a neural network    '''


    def __init__(self, layers, loss_func, learn_rate):
        self.layers = []
        self.loss_func = 'square_error'
        self.learn_rate = 0.2
        raise NotImplementedError("SequentialModel")

    def add_layer(self, layer, pos=-1):
        '''Adds a new layer `layer` to the model at position pos in sequence'''
        raise NotImplementedError("add_layer")

    def train(self, x_data, y_data, epochs):
        '''Goes through each input datapoint in x_data and y_data and perform feed_forward and feed_backward on each of the datapoint. Repeat the procedure for epochs times.'''
        raise NotImplementedError("train")

    def predict(self, x_data):
        '''Performs predictions on all datapoints in x_data and returns an array of the predictions given by the results of feed_forward'''
        raise NotImplementedError("predict")

    def save(self, filename):
        '''Saves all layers and hyperparameters into an yaml file (using the yaml library)'''
        raise NotImplementedError("save")

    def load(self, filename):
        '''Loads all layers and hyperparameters from an yaml file (using the yaml library)'''
        raise NotImplementedError("load")
class DenseLayer(object):


    def __init__(self, units, activation):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None

    def apply(self, x_data):
        input_size = x_data.shape[0]
        if self.weights is None:
            self.weights = np.random.rand(input_size, self.units)
            self.bias = np.zeros(self.units)
        z = np.dot(x_data, self.weights) + self.bias
        if self.activation == 'relu':
            a = np.maximum(z, 0)
        elif self.activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z))
            a = exp_z / np.sum(exp_z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return a
    def back_propagation(self, x_data, y_real, y_pred, learn_rate):
        raise NotImplementedError("feed_forward")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    iris_data = pd.read_csv('iris.csv')
    X = iris_data.iloc[:, :-1].values
    y = iris_data.iloc[:, -1].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )
    denseLayer = DenseLayer(3, 'relu' )
    denseLayer.apply(X_train[0])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
