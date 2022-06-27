# https://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

import numpy as np
from Models.sequential import Sequential
from Losses.losses import d_MSE

def GD(loss_fn, model, labels, preds):
    assert isinstance(model, Sequential), "Model type not recognized"

    loss = loss_fn(labels, preds)

    # loss_name = loss_fn.__name__ # TODO: Find the derivative

    for i in range(len(model.layers)-1, -1, -1): # Start from last layer
        layer = model.layers[i]

        for j in layer[j].weights: # Select the weights from neurons of the actual layer
            # Compute the partial derivative of the cost with respect to this weight
            pass