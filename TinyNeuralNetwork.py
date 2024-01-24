import numpy as np

class TinyNeuralNetwork:
    def __init__(self, input, output, hidden_layer_sizes=None, weights_biases=None):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = []
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

    def backpropagation(self, expected_outputs, learning_rate=.5):
        # output layer deltas
        out = self.layers[-1]
        error = -(expected_outputs - out)
        delta1 = out * (1.0 - out) * error

        for i, w, b in self.rev_weights_biases:
            a0 = self.layers[i]
            change_w = np.multiply.outer(a0, delta1)
            w -= learning_rate * change_w  # changes w in place
            b -= learning_rate * delta1  # changes b in place

            if i:  # Skip delta on input nodes since we don't use it (and it is 0)
                error = np.dot(w, delta1)
                delta1 = a0 * (1.0 - a0) * error


    def train(self, training_data, batch_size, num_training_iterations, learning_rate=.5):
        # clean data
        training_inputs = np.array([np.array(i) for i, o in training_data])
        training_outputs = np.array([np.array(o) for i, o in training_data])
        training_data_size = len(training_data)

        # make functions local
        layers = self.layers
        weights_and_biases = self.weights_and_biases
        rev_weights_biases = self.rev_weights_biases

        # loop
        for indx in range(num_training_iterations):
            # create a random batch
            indices = np.random.choice(training_data_size, batch_size)
            # indices = [indx % training_data_size]
            inputs = training_inputs[indices]
            expected_output = training_outputs[indices]

            # feed forward, vectorized over whole batch
            a = inputs
            layers[0] = a

            for i, w, b in weights_and_biases:
                sum = np.einsum('jk,ij->ik', w, a) + b  # np.matmul(a, w) + b
                a = .5 * (1 + np.tanh(.5 * sum)) #1.0 / (1.0 + exp(-sum)) # To avoid overflow
                layers[i + 1] = a

            # back propogate, vectorized over whole batch
            out = layers[-1]
            error = -(expected_output - out)
            delta1 = out * (1.0 - out) * error

            for i, w, b in rev_weights_biases:
                a0 = layers[i]
                change_w = np.einsum('ij,ik->jk', a0, delta1)
                change_b = np.einsum('ik->k', delta1)
                w -= learning_rate * change_w / batch_size  # changes w in place
                b -= learning_rate * change_b / batch_size  # changes b in place

                if i:  # Skip delta on input nodes since we don't use it (and it is 0)
                    error = np.einsum('jk,ik->ij', w, delta1)  # dot(w, delta1)
                    delta1 = a0 * (1.0 - a0) * error


    def export_model(self):
        import pickle
        dump = pickle.dumps(self)
        import zlib
        compressed_dump = zlib.compress(dump)
        import base64
        return base64.b64encode(compressed_dump)

    def info(self):
        print("layers:")
        for l in self.layers:
            print(l)
        print("weights:")
        for w in self.weights:
            print(w)
        print("biases:")
        for b in self.biases:
            print(b)

def import_model( model):
    import pickle
    import zlib
    import base64
    dump = base64.b64decode(model)
    dump = zlib.decompress(dump)
    return pickle.loads(dump)
