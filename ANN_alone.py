import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        #Equivalent sigmoid expression to avoid OverFlow
        # Source - https://stackoverflow.com/a
        # Posted by DYZ, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-11-16, License - CC BY-SA 4.0
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def tanh(x): return np.tanh(x)
    @staticmethod
    def linear(x): return x
    @staticmethod
    def input(x):return x
class Layer:
    # Initialize random weights and bias
    def __init__(self, n_input, n_output, activation):

        # Weight representation in self.W :
        #
        # n_input \ n_output  Neuron1   Neuron2  ...        Neuron n_input
        # Weight 1              .          .                    .
        # Weight 2              .           .   ...             .           
        #   ...
        # Weight n_output       .           .                   .

        # n_input : number of input for each neuron ; 
        # n_output : number of neurons
        self.n_input = n_input
        self.n_neuron = n_output
        self.W = np.random.randn(n_input, n_output) # “standard normal” distribution
        self.b = np.random.randn(n_output)
        self.activationId= activation
        self.activation = getattr(Activation, activation)

    def __eq__(self, value): 

        assert(self.W.shape == value.W.shape) # error : unmatched dimensions

        if ((self.W == value.W).all()) and ((self.b == value.b).all()) :
            # if equal the resulting vector is null
            return True 
        
        return False
    
    def copy(self):
        # == AI == 
        #Prompt : given the clas Layer implements a copy method
        clone = Layer(self.n_input, self.n_neuron, self.activationId)
        clone.W = np.copy(self.W)
        clone.b = np.copy(self.b)
        return clone
    
# ==        ANN         ==
# Hyperparameters
# number of hidden layer : depends on the problem complexity
# number of neurons per hidden layer : affects the generalisation (undefitting VS overfitting)
# activation functions : since we have a regression problem, 
# it is convenient to have a linear activation function in the output and others function in the hidden layer (lecture 2)

class ANN:
    @property
    def layer_size(self): 
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
        assert len(layer_sizes) == len(activations), \
            "the total of activation functions must be equal to the number of hidden layers + output layer"


        self.layer_sizes = layer_sizes # [dimension of the input, size of hidden layer 1, ..., size of hidden layer n, size of output layer]
        self.activations = activations # e.g ['input',..., 'relu']

        # Create successive layers
        # [hidden layer 1, hidden layer 2 , ..., output]
        self.layers = [
            #     n_input           n_output          activation function
            Layer(layer_sizes[i-1], layer_sizes[i], activations[i])
            for i in range(1,len(layer_sizes))
        ]

    def __eq__(self, ANN2):
        
        if (self.get_params() == ANN2.get_params()).all():
            return True 
        return False
    

    def copy(self):
        clone = ANN(layer_sizes=self.layer_sizes, activations= self.activations)
        l = 0
        for layer in self.layers : 
            clone.layers[l] = layer.copy()
            l+=1
    
        return clone

    
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
        return np.array(params).astype('float')

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
              activations=['input',"relu", "tanh"])

    # Random input data (e.g., 3 samples, 8 features)
    X = np.random.rand(3, 8) #??

    # Forward propagation
    Y_pred = ann.forward(X)

    print("Input :\n", X)
    print("\nPredicted Output :\n", Y_pred)

    # Check size of the parameter vector
    params = ann.get_params()
    print("\nSize of parameter vector :", len(params))
