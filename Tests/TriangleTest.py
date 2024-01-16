import random
import time
import Utils

import TinyNeuralNetwork as tnn

#  Given a point P(x, y) and a triangle formed by points A, B, and C, determine if P is within triangle ABC.

#  The solution is based on the barycentric coordinates.
def point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
               (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(p, a, b) < 0.0
    b2 = sign(p, b, c) < 0.0
    b3 = sign(p, c, a) < 0.0

    return ((b1 == b2) and (b2 == b3))

def random_dataset_of_triangles_and_points(dataset_size):
    new_list = []
    for i in range(dataset_size):
        # generate 8 floats
        random_floats = [random.random() for _ in range(8)]
        in_triangle = point_in_triangle(random_floats[0:2], random_floats[2:4], random_floats[4:6], random_floats[6:8])
        new_list.append((random_floats, Utils.bool_to_bits(in_triangle)))
    return new_list


testCount = 100
trainingCount = 10000
training_data = random_dataset_of_triangles_and_points(trainingCount)
test_data = random_dataset_of_triangles_and_points(testCount)

# 1 bit output : is the point in the triangle ?
neural_network = tnn.TinyNeuralNetwork(8, 1, [20, 20])

print("Training...")
start_time = time.perf_counter()
neural_network.train_in_batches(training_data, 1024, 10000, 0.5)
print("Training took: ", time.perf_counter() - start_time)


# Test network
testResults = []
for i in range(testCount):
    output = neural_network.get_output(test_data[i][0])
    int_output = round(output[0])
    int_expected_output = Utils.bits_to_bool(test_data[i][1])
    print("Input: ", test_data[i][0], " Expected output: ", int_expected_output, " Output: ", int_output, " Success: ", int_output == int_expected_output)

    testResults.append(int_output == int_expected_output)


# % of success
successRate = sum(testResults) / testCount
print("Success rate: ", successRate * 100, "%")

