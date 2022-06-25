# https://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

import numpy as np
from Models.sequential import Sequential

def GD(loss_fn, model, labels, preds):
    assert isinstance(model, Sequential), "Model type not recognized"

    for i in range(len(model.layers)-1, -1, -1):
        layer = model.layers[i]
        for j in range(len(layer.weights)):
            weight = layer.weights[j]
            print(weight)
        break
