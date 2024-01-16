# Tiny Neural Network

This is a tiny neural network implementation in Python3/NumPy.

## Usage

Paste the code into your project and use it as follows:

```python
import TinyNeuralNetwork as tnn

neural_network = tnn.TinyNeuralNetwork(inputLayers, outputLayers, hiddenLayers)
# first, train the network
neural_network.train_in_batches(training_data, batchSize, trainingIterations, learningRate)
# then, use the network
output = neural_network.get_output(test_data)
```
## Export model

In order to properly export the model, you need to save the weights and biases of the network.

First, avoid module import by pasting your training code at the end of the file TinyNeuralNetwork.py. 

Then, add the following code at the end of your training code:

```python
#export model
import pickle
dump = pickle.dumps(neural_network)
import zlib
compressed_dump = zlib.compress(dump)
import base64
str = base64.b64encode(compressed_dump)
print("Exported model:")
print(str)
```

## Import model

To import the model, you need to paste the code of the TinyNeuralNetworkLite.py (without all training stuff) file into your project and then use the following code:

```python
# import model
model = b'YOUR_MODEL_STRING'
neural_network_loaded = import_model(model) 
```
Then you can use the network as usual.

## Hyperparameters tweaking

The hyperparameters are the following:

- inputLayers: number of input neurons
- outputLayers: number of output neurons
- hiddenLayers: number of hidden neurons
- batchSize: number of training samples per batch
- trainingIterations: number of training iterations
- learningRate: learning rate


