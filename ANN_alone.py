import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def tanh(x): return np.tanh(x)

class Layer:
    # Initialize random weights and bias
    def __init__(self, n_input, n_output, activation):
        self.W = np.random.randn(n_input, n_output)
        self.b = np.random.randn(n_output)
        self.activation = getattr(Activation, activation)

class ANN:
    @property
    def layer_size(self): #???
        """
        Returns the list of sizes of each layer (input, hidden layers..., output).
        Infers sizes from weights if the original attribute was not stored.
        """
        # If the original sizes were stored elsewhere, prefer that
        if hasattr(self, "_layer_sizes") and self._layer_sizes is not None:
            return list(self._layer_sizes)

        # Infer sizes from layers if available
        if hasattr(self, "layers") and self.layers:
            sizes = [self.layers[0].W.shape[0]]
            for layer in self.layers:
                sizes.append(layer.W.shape[1])
            return sizes

        # Nothing available yet
        return []
    
    def __init__(self, layer_sizes, activations):
        assert len(layer_sizes) - 1 == len(activations), \
            "the total of activation functions must be equal to the number of hidden layers + output layer"

        self.layer_sizes = layer_sizes
        # Create successive layers
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = np.dot(x, layer.W) + layer.b
            x = layer.activation(x)
        return x

    # Returns a 1D numpy array of all weights and biases in the network
    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.W.flatten())
            params.extend(layer.b.flatten())
        return np.array(params)

    # Sets weights and biases from a 1D numpy array
    def set_params(self, params):
        idx = 0
        for layer in self.layers:
            w_size = layer.W.size
            b_size = layer.b.size

            layer.W = params[idx:idx + w_size].reshape(layer.W.shape)
            idx += w_size
            layer.b = params[idx:idx + b_size].reshape(layer.b.shape)
            idx += b_size


# Example
if __name__ == "__main__":
    # Network: 8 inputs → 5 hidden neurons → 1 output
    ann = ANN(layer_sizes=[8, 5, 1],
              activations=["relu", "tanh"])

    # Random input data (e.g., 3 samples, 8 features)
    X = np.random.rand(3, 8)

    # Forward propagation
    Y_pred = ann.forward(X)

    print("Input :\n", X)
    print("\nPredicted Output :\n", Y_pred)

    # Check size of the parameter vector
    params = ann.get_params()
    print("\nSize of parameter vector :", len(params))
