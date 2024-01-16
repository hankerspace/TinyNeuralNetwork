import TinyNeuralNetwork as tnn
import random
import matplotlib.pyplot as plt

def xor(list):
    sum = 0
    for i in list:
        sum += i
    return sum % 2

def random_set_of(listSize, bits, generationFunction):
    new_list = []
    for i in range(listSize):
        random_list = [random.randrange(2) for _ in range(bits)]
        new_list.append((random_list, [generationFunction(random_list)]))
    return new_list

testCount = 1000
trainingCount = 100000
training_data = random_set_of(trainingCount, 4, xor)
test_data = random_set_of(testCount, 4, xor)

chart_data = []
for batch_size in (16, 32, 64, 128, 256, 512, 1024, 2048):
    for num_traning_iterations in (100, 200, 500, 1000, 2000, 5000, 10000):
        for learning_rate in (0.1, 0.2, 0.3, 0.4, 0.5):
            neural_network = tnn.TinyNeuralNetwork(4, 1, [20, 20])
            neural_network.train_in_batches(training_data, batch_size, num_traning_iterations, learning_rate)
            testResults = []
            for i in range(testCount):
                output = neural_network.get_output(test_data[i][0])
                testResults.append(round(output[0]) == test_data[i][1][0])
            successRate = sum(testResults) / testCount
            print("Batch size: ", batch_size, " Num training iterations: ", num_traning_iterations, " Learning rate: ", learning_rate, " Success rate: ", successRate * 100, "%")
            chart_data.append((batch_size, num_traning_iterations, learning_rate, successRate))

    # draw result chart (num training iterations, learning rate, success rate)
    x = []
    y = []
    z = []
    for i in chart_data:
        if i[0] == batch_size:
            x.append(i[1])
            y.append(i[2])
            z.append(i[3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('Num training iterations')
    ax.set_ylabel('Learning rate')
    ax.set_zlabel('Success rate')
    ax.set_title('Batch size: ' + str(batch_size))
    plt.show()




