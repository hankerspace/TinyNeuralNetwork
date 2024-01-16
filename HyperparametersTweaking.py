import TinyNeuralNetwork as tnn
import matplotlib.pyplot as plt

from Tests.DatasetGenerationFunction import random_dataset_of_bits, xor

testCount = 1000
trainingCount = 100000
training_data = random_dataset_of_bits(trainingCount, 4, xor)
test_data = random_dataset_of_bits(testCount, 4, xor)

chart_data = []
learning_rate = 0.5
for batch_size in (32, 64, 128, 256, 512):
    for num_traning_iterations in (500, 1000, 2000, 5000, 10000, 20000):
        neural_network = tnn.TinyNeuralNetwork(4, 1, [20, 20])
        neural_network.train_in_batches(training_data, batch_size, num_traning_iterations, learning_rate)
        testResults = []
        for i in range(testCount):
            output = neural_network.get_output(test_data[i][0])
            testResults.append(round(output[0]) == test_data[i][1][0])
        successRate = sum(testResults) / testCount
        print("Batch size: ", batch_size, " Num training iterations: ", num_traning_iterations, " Learning rate: ", learning_rate, " Success rate: ", successRate * 100, "%")
        chart_data.append((batch_size, num_traning_iterations, learning_rate, successRate))

# draw result chart (num training iterations, success rate)
# create one serie per batch size
series = {}
for batch_size, num_traning_iterations, learning_rate, successRate in chart_data:
    if not batch_size in series:
        series[batch_size] = ([], [])
    series[batch_size][0].append(num_traning_iterations)
    series[batch_size][1].append(successRate)

# draw chart
plt.figure(figsize=(10, 5))
plt.title("Success rate depending on number of training iterations")
plt.xlabel("Number of training iterations")
plt.ylabel("Success rate")
for batch_size in series:
    plt.plot(series[batch_size][0], series[batch_size][1], label="Batch size: " + str(batch_size))
plt.legend()
plt.show()





