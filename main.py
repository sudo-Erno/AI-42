from Layers.linear_perceptron import Linear
from Models.sequential import Sequential
import Losses.losses as Loss
from Optimizers.GD import GD
import numpy as np

# Smokes    Exercise    Fat

test_inputs = [[0, 1, 0], [1, 0, 1], [0, 1, 1]]
test_labels = [0, 1, 0]

test_inputs = np.array(test_inputs)
test_labels = np.array(test_labels)

l1 = Linear(1, activation="ReLU")
l2 = Linear(1, activation="ReLU")
l3 = Linear(1, activation="ReLU")

model = Sequential([
    l1,
    l2,
    l3
])

y_pred = model.forward(test_inputs, test_labels, epochs=1)
# loss = Loss.MSE(test_labels, y_pred)
print(y_pred)

GD(Loss.MSE, model, test_labels, y_pred)