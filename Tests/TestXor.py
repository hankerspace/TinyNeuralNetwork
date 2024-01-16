import TinyNeuralNetwork as tnn
import time

from Tests.DatasetGenerationFunction import xor, random_dataset_of_bits

input_bits = 4
testCount = 100
trainingCount = 10000
training_data = random_dataset_of_bits(trainingCount, input_bits, xor)
test_data = random_dataset_of_bits(testCount, input_bits, xor)


neural_network = tnn.TinyNeuralNetwork(input_bits, 1, [20, 20])

start_time = time.perf_counter()
neural_network.train_in_batches(training_data, 256, 10000, 0.5)
print("Training took: ", time.perf_counter() - start_time)


# Test network
testResults = []
for i in range(testCount):
    output = neural_network.get_output(test_data[i][0])
    testResults.append(round(output[0]) == test_data[i][1][0])

# % of success
successRate = sum(testResults) / testCount
print("Success rate: ", successRate * 100, "%")

#export model
import pickle
dump = pickle.dumps(neural_network)
import zlib
compressed_dump = zlib.compress(dump)
import base64
str = base64.b64encode(compressed_dump)
print("Exported model:")
print(str)