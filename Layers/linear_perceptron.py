import Layers.activation_functions as activations_fn

class Linear:
    def __init__(self, number, input_shape=None, activation=None):
        self.number = number
        self.input_shape = input_shape
        self.activation = activation
        self.weights = None
        
        if activation == "ReLU":
            self.activation = activations_fn.relu
        elif activation == "Sigmoid":
            self.activation = activations_fn.sigmoid
        elif activation == "Tanh":
            self.activation = activations_fn.tanh
        elif activation == "leakyRelu":
            self.activation = activations_fn.leakyRelu