import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.linalg.norm(W)**2
    grad = reg_strength * 2 * W

    return loss, grad


def softmax(predictions):
    pred = predictions.copy()
    if predictions.ndim == 1:
        pred -= np.max(pred)
        return np.exp(pred) / np.sum(np.exp(pred))
    else:
        pred -= np.max(predictions, axis=1).reshape(-1, 1)
        return np.exp(pred)/np.sum(np.exp(pred), axis=1).reshape(-1, 1)
    

def cross_entropy_loss(probs, target_index):
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    else:
        loss = 0
        
        for i in range(probs.shape[0]):
            loss -= np.log(probs[i, target_index[i]])
      
        return loss/probs.shape[0]


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    s = softmax(predictions)
    loss = cross_entropy_loss(s, target_index)

    if predictions.ndim == 1:
        s[target_index] -= 1
    else:
        for i in range(s.shape[0]):
            s[i, target_index[i]] -= 1
        s /= s.shape[0]
    return loss, s


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        pred = np.maximum(0, X)
        return pred

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out * (self.X >= 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        pred = np.dot(X, self.W.value) + self.B.value
        return pred

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0)[:, np.newaxis].T
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height + 2*self.padding - self.filter_size + 1
        out_width = width + 2*self.padding - self.filter_size + 1
        
        X_pad = np.zeros((batch_size, height+2*self.padding, width+2*self.padding, channels))
        X_pad[:, self.padding:self.padding+height, self.padding:self.padding+width, :] = X

        self.X_forward = (X, X_pad)

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                window = X_pad[:, y:y+self.filter_size, x:x+self.filter_size, :, np.newaxis]
                out[:, y, x, :] = np.sum(window*self.W.value, axis = (1, 2, 3)) + self.B.value
        return out


    def backward(self, d_out):
        X, X_pad = self.X_forward
        X_grad = np.zeros(X_pad.shape)

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        for y in range(out_height):
            for x in range(out_width):
                window = X_pad[:, y:y+self.filter_size, x:x+self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * window, axis=0)
                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=4)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
        X_grad_without_pad = X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]
        return X_grad_without_pad

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                window = X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                out[:, y, x, :] = np.max(window, axis=(1,2))

        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros(self.X.shape)

        for y in range(out_height):
            for x in range(out_width):
                window = self.X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                grad = d_out[:, y, x, :]
                max_elements = (window == np.max(window, axis=(1,2))[:, np.newaxis, np.newaxis, :])
                grad = grad[:, np.newaxis, np.newaxis, :]

                out[:, y:y+self.pool_size, x:x+self.pool_size, :] += grad*max_elements
        return out


        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height*width*channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}