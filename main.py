from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import yaml

class SequentialModel(object):
    '''A class that manages a certain sequence of layers in a neural network    '''


    def __init__(self, layers, loss_func, learn_rate):
        self.layers = list(layers)
        self.loss_func = loss_func
        self.learn_rate = learn_rate

    def add_layer(self, layer, pos=-1):
        '''Adds a new layer `layer` to the model at position pos in sequence'''
        if pos == -1:
            self.layers.append(layer)
        else:
             self.layers.insert(pos, layer)

    def train(self, x_data, y_data, epochs):
        '''Goes through each input datapoint in x_data and y_data and perform feed_forward and feed_backward on each of the datapoint. Repeat the procedure for epochs times.'''
        for epoch in range(epochs):
            losses = []
            for i in range(len(x_data)):
                x_sample = x_data[i]
                y_sample = y_data[i]
                y_pred = self.predict(x_sample)
                mse = np.mean((y_sample - y_pred) ** 2)
                losses.append(mse)
                grad = y_pred - y_sample
                for layer in reversed(self.layers):
                    grad = layer.back_propagation(grad, self.learn_rate)
            epoch_loss = np.mean(losses)
            print(epoch_loss)

    def predict(self, x_data):
        '''Performs predictions on all datapoints in x_data and returns an array of the predictions given by the results of feed_forward'''
        output = x_data
        for layer in self.layers:
            output = layer.apply(output)
        return output

    def save(self, filename):
        '''Saves all layers and hyperparameters into an yaml file (using the yaml library)'''

        data = {
            'learn_rate': self.learn_rate,
            'loss_function': self.loss_func,
            'layers': []
        }

        for layer in self.layers:
            data['layers'].append({
                'units': layer.units,
                'activation': layer.activation,
                'weights': layer.weights.tolist(),
                'bias': layer.bias.tolist()
            })

        with open(filename, 'w') as file_out:
            yaml.dump(data, file_out)

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
        self.last_input = x_data
        self.last_z = z
        self.last_output = a
        return a
    def back_propagation(self, grad, learn_rate):
        if self.activation == 'relu':
            activation_grad = (self.last_z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-self.last_z))
            activation_grad = sigmoid * (1 - sigmoid)
        elif self.activation == 'softmax':
            activation_grad = self.last_output * (1 - self.last_output)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        # Local gradient
        grad = np.multiply(activation_grad, grad)
        weight_grad = np.outer(self.last_input, grad)
        bias_grad = grad
        prev_grad = np.dot(self.weights, grad)
        self.weights = self.weights - learn_rate * weight_grad
        self.bias = self.bias - learn_rate * bias_grad
        return prev_grad

if __name__ == '__main__':
    np.random.seed(42)
    iris_data = pd.read_csv('iris.csv')
    X = iris_data.iloc[:, :-1].values
    y = iris_data.iloc[:, -1].values

    label_map = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2
    }

    y_int = np.array([label_map[label] for label in y])

    X_train, X_temp, y_train_int, y_temp_int = train_test_split(
        X, y_int,
        test_size=0.2,
        stratify=y_int,
        random_state=42
    )

    X_val, X_test, y_val_int, y_test_int = train_test_split(
        X_temp, y_temp_int,
        test_size=0.5,
        stratify=y_temp_int,
        random_state=42
    )

    y_train = np.eye(3)[y_train_int]
    y_val = np.eye(3)[y_val_int]
    y_test = np.eye(3)[y_test_int]
    model = SequentialModel([], 'square_error', 0.2)
    model.add_layer(DenseLayer(5, 'relu'))
    model.add_layer(DenseLayer(3, 'softmax'))

    sample = X_train[0]
    pred = model.predict(sample)

    model.train(X_train, y_train, 10)
    correct = 0

    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        pred_class = np.argmax(pred)
        true_class = np.argmax(y_test[i])

        if pred_class == true_class:
            correct += 1

    accuracy = correct / len(X_test)
    print("Accuracy:", accuracy)
    model.save('model.yaml')



