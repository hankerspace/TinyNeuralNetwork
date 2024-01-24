# In this example we will create a neural network which determines if a point is inside ou outside a given circle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def generate_data(num_samples):
    centers = np.random.rand(num_samples, 2)
    radii = np.random.rand(num_samples, 1)
    points = np.random.rand(num_samples, 2)

    labels = np.linalg.norm(points - centers, axis=1) <= radii[:, 0]
    labels = labels.astype(int)

    inputs = np.hstack((centers, radii, points))

    return inputs, labels

# Training
X_train, y_train = generate_data(50000)

model = Sequential([
    Dense(64, activation='sigmoid', input_dim=5), # Be careful, we only use sigmoid here because we want to export the model to TinyNeuralNetwork
    Dense(32, activation='sigmoid'), # Be careful, we only use sigmoid here because we want to export the model to TinyNeuralNetwork
    Dense(1, activation='sigmoid') # Be careful, we only use sigmoid here because we want to export the model to TinyNeuralNetwork
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=1024)

# Testing
X_test, y_test = generate_data(1000)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

X_new, y_new = generate_data(1)
prediction =  round(model.predict(X_new)[0][0])
print(f"Prediction: {prediction}, Expected: {y_new[0]}")

############################################################################################################
# Exporting the model to TinyNeuralNetwork

# Create a new TinyNeuralNetwork with the same architecture
import TinyNeuralNetwork as tnn

# Exporting weights and biases
weights = []
biases = []
for layer in model.layers:
    weights.append(layer.get_weights()[0])
    biases.append(layer.get_weights()[1])

tiny_neural_network = tnn.TinyNeuralNetwork(5, 1, [64, 32], (weights, biases))

# Test the new network

X_new, y_new = generate_data(100)
results = []
for i in range(len(X_new)):
    prediction = tiny_neural_network.get_output(X_new[i])[0]
    #print(f"Prediction from TNN: {prediction} ({round(prediction)}, Expected: {y_new[i]}")
    results.append(round(prediction) == y_new[i])
print(f"TNN Success rate: {sum(results) / len(results)}")

# Now you can export the TNN which was trained with TensorFlow