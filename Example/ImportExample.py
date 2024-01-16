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


from Tests.DatasetGenerationFunction import random_dataset_of_bits, xor

input_bits = 4
testCount = 100
test_data = random_dataset_of_bits(testCount, input_bits, xor)

# load model (from xor training)
model = b'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'
neural_network2 = import_model(model)

# Test network
testResults = []
for i in range(testCount):
    output = neural_network2.get_output(test_data[i][0])
    testResults.append(round(output[0]) == test_data[i][1][0])

# % of success
successRate = sum(testResults) / testCount
print("Success rate: ", successRate * 100, "%")