import numpy as np
import pandas as pd

class SequentialModel(object):
    '''A class that manages a certain sequence of layers in a neural network    '''
    self.layers = []
    self.loss_func = 'square_error'
    self.learn_rate = 0.2

    def __init__(self, layers, loss_func, learn_rate):
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
    self.units = 1
    self.activation = 'relu'
    self.weights = None

    def __init__(self, units, activation):
        raise NotImplementedError("DenseLayer")

    def apply(self, x_data):
        raise NotImplementedError("apply")

    def back_propagation(self, x_data, y_real, y_pred, learn_rate):
        raise NotImplementedError("feed_forward")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Hello World")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
