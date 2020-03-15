import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net
    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network
        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        _, _, input_channels = input_shape

        self.conv1 = ConvolutionalLayer(input_channels, conv1_channels, 3, 1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolingLayer(4, 4)
        
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolingLayer(4, 4)

        self.flattener = Flattener()
        self.fc = FullyConnectedLayer(4*conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for param_ix in self.params():
            self.params()[param_ix].grad = np.zeros_like(self.params()[param_ix].value)

        out = self.pool1.forward(self.relu1.forward(self.conv1.forward(X)))
        out = self.pool2.forward(self.relu2.forward(self.conv2.forward(out)))
        out = self.fc.forward(self.flattener.forward(out))

        loss, grad = softmax_with_cross_entropy(out, y)

        grad = self.flattener.backward(self.fc.backward(grad))
        grad = self.conv2.backward(self.relu2.backward(self.pool2.backward(grad)))
        grad = self.conv1.backward(self.relu1.backward(self.pool1.backward(grad)))

        return loss

    def predict(self, X):
        out = self.pool1.forward(self.relu1.forward(self.conv1.forward(X)))
        out = self.pool2.forward(self.relu2.forward(self.conv2.forward(out)))
        out = self.fc.forward(self.flattener.forward(out))

        predictions = softmax(out)
        return np.argmax(predictions, axis=1)

    def params(self):
        result = {'conv1.W': self.conv1.W, 'conv1.B': self.conv1.B,
                'conv2.W': self.conv1.W, 'conv2.B': self.conv1.B,
                'fc.W': self.fc.W, 'fc.B': self.fc.B}

        return result