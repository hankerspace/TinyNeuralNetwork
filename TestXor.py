import NeuralNetworkFull as nn
import random
import time

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

testCount = 100
trainingCount = 10000
training_data = random_set_of(trainingCount, 9, xor)
test_data = random_set_of(testCount, 9, xor)

neural_network = nn.NeuralNetwork(9, 1, [20, 20])

start_time = time.perf_counter()
neural_network.train_in_batches(training_data, 1000, 10000, 0.3)
print("Training took: ", time.perf_counter() - start_time)


# Test network
testResults = []
for i in range(testCount):
    output = neural_network.get_output(test_data[i][0])
    testResults.append(round(output[0]) == test_data[i][1][0])

# % of success
print(sum(testResults) / testCount)




