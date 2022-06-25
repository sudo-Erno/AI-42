import numpy as np
from Layers.linear_perceptron import Linear

class Sequential():
    def __init__(self, layers):
        self.layers = layers

        if not isinstance(layers, list):
            print("Parameter layers must be a list")
            exit()

        for i in range(len(layers)):
            if not isinstance(layers[i], Linear):
                print(f"Layer {i} not found")
                quit()

        # Create weights

        for i in range(len(layers) - 1):
            layer = layers[i]
            next_layer = layers[i + 1]
            layers[i+1].weights = np.ones((next_layer.number, layer.number))

    def forward(self, input_data, labels, epochs=1):
        input_size = 0
        
        if isinstance(input_data, np.ndarray):
            input_size = input_data.shape[0]
        
        elif isinstance(input_data, list):
            input_size = len(input_data)
        
        else:
            print("Wrong data type for input")
            quit()

        # Add the weights between input and first layer
        first_weights = np.ones((self.layers[0].number, input_size))
        
        self.layers[0].weights = first_weights

        # Feedforward
        for _ in range(epochs):
            z = input_data

            for layer in self.layers:
                
                weights = layer.weights
                activation_function = layer.activation

                z = np.matmul(weights, z)

                if activation_function:
                    z = activation_function(z)

        return z
    
