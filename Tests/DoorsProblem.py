import random
import time
import Utils

import TinyNeuralNetwork as tnn

# The 100 Doors Problem : https://rosettacode.org/wiki/100_doors#Python
def compute_doors_output(number_of_doors):
    doors = [False] * number_of_doors
    for i in range(number_of_doors):
       for j in range(i, number_of_doors, i+1):
           doors[j] = not doors[j]
    # count open doors
    count = 0
    for i in range(number_of_doors):
        if doors[i]:
            count += 1
    # return 8 bits integer (0-255)
    return Utils.int_to_bits(count, 8)

def random_dataset_of_doors(dataset_size):
    new_list = []
    for i in range(dataset_size):
        random_int = [random.randrange(2) for _ in range(8)] #8 bits integer (0-255)
        integer = Utils.bits_to_int(random_int)
        new_list.append((random_int, compute_doors_output(integer)))
    return new_list


# Lets try to solve the doors problem with a neural network. Instead of 100 doors, we will use random 0-255 doors (8 bits integer).
# There are 100 doors in a row that are all initially closed.
# You make 100 passes by the doors.
# The first time through, visit every door and  toggle  the door  (if the door is closed,  open it;   if it is open,  close it).
# The second time, only visit every 2nd door   (door #2, #4, #6, ...),   and toggle it.
# The third time, visit every 3rd door   (door #3, #6, #9, ...), etc,   until you only visit the 100th door.

testCount = 100
trainingCount = 10000
training_data = random_dataset_of_doors(trainingCount)
test_data = random_dataset_of_doors(testCount)

# 8 bits input : number of doors
# 8 bits output : number of open doors
neural_network = tnn.TinyNeuralNetwork(8, 8, [20, 20])

start_time = time.perf_counter()
neural_network.train(training_data, 256, 10000, 0.5)
print("Training took: ", time.perf_counter() - start_time)


# Test network
testResults = []
for i in range(testCount):
    int_input = Utils.bits_to_int(test_data[i][0])
    output = neural_network.get_output(test_data[i][0])
    # round output to 0 or 1
    rounded_output = []
    for j in range(len(output)):
        rounded_output.append(round(output[j]))
    # compute int output
    int_output = Utils.bits_to_int(rounded_output)
    int_expected_output = Utils.bits_to_int(test_data[i][1])

    testResults.append(int_output == int_expected_output)

    print("Input: ", int_input, " Expected output: ", int_expected_output, " Output: ", int_output, " Success: ", int_output == int_expected_output)

# % of success
successRate = sum(testResults) / testCount
print("Success rate: ", successRate * 100, "%")
