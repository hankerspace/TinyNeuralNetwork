import numpy as np

class TinyNeuralNetwork:
    def __init__(self, input, output, hidden_layer_sizes=[], weights_biases=None):
        self.input = input  # number of input nodes + 1 for bias
        self.output = output  # number of output nodes

        # Layers
        self.layer_sizes = [self.input] + list(hidden_layer_sizes) + [self.output]
        self.layers = [np.ones(s) for s in self.layer_sizes]
        self.num_layers = len(self.layers)

        # Weights and biases
        if weights_biases is None:
            self.weights = []
            self.biases = []
            for s0, s1 in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
                self.weights.append(np.random.randn(s0, s1))
                self.biases.append(np.random.randn(s1))
        else:
            self.weights, self.biases = weights_biases

        # Lists for enumeration
        self.weights_and_biases = list(zip(range(self.num_layers - 1), self.weights, self.biases))
        self.rev_weights_biases = list(reversed(self.weights_and_biases))

    def get_output(self, inputs):
        # set inputs
        a = inputs
        self.layers[0] = a

        for i, w, b in self.weights_and_biases:
            sum = np.dot(w.T, a) + b
            a = 1 / (1 + np.exp(-sum))
            self.layers[i + 1] = a

        return self.layers[-1]

def import_model( model):
    import pickle
    import zlib
    import base64
    dump = base64.b64decode(model)
    dump = zlib.decompress(dump)
    return pickle.loads(dump)
