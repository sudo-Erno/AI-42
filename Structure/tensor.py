from collections import defaultdict
from turtle import xcor
import numpy as np

class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, y):
        value = self.value + y.value
        local_gradients = (
            (self, 1),
            (y, 1)
        )
        return Variable(value, local_gradients)
    
    def __mul__(self, y):
        value = self.value * y.value
        local_gradients = (
            (self, y.value),
            (y, self.value)
        )
        return Variable(value, local_gradients)

    def __matmul__(self, y):
        assert isinstance(y, np.ndarray), "Matrix must be numpy"
        
        self_rows, self_cols = self.shape
        y_rows, y_cols = y.shape
        
        assert self_cols == y_rows, f"Matrix A rows ({self_rows}) don't match with Matrix B cols ({y_cols})"

        new_mat = []

        # O(i * j * k)....very expensive....
        
        for i in range(y_cols):
            for j in range(self_rows):
                c = 0
                for k in range(self_cols):
                    print(type(w_mat[j][k]))
                    c += w_mat[j][k] * x_mat[i][k]
                new_mat.append(c)

        new_mat = np.array(new_mat)
        new_mat = np.reshape(new_mat, (self_rows, y_cols))

        return new_mat

    def get_gradients(self):

        gradients = defaultdict(lambda: 0)
        
        def compute_gradients(tensor, path_value):
            for child_tensor, local_gradient in tensor.local_gradients:

                value_of_path_to_child = path_value * local_gradient

                gradients[child_tensor] += value_of_path_to_child

                compute_gradients(child_tensor, value_of_path_to_child)

        compute_gradients(self, 1)

        return gradients

def to_Variable(x):

    assert isinstance(x, np.ndarray), "Variable must be a numpy array"
    
    rows, cols = x.shape
    new_mat = []
    
    for i in range(rows):
        for j in range(cols):
            new_mat.append(Variable(x[i][j]))
    
    new_mat = np.array(new_mat)
    # new_mat = Variable(np.reshape(new_mat, (rows, cols)))
    new_mat = np.reshape(new_mat, (rows, cols))

    return new_mat

if __name__ == '__main__':

    w_mat = np.array([[1, 1, 1], [1, 1, 1]])
    x_mat = np.array([[1], [1], [1]])

    # Convert elements of numpy array to Variable type
    w_mat = to_Variable(w_mat)
    x_mat = to_Variable(x_mat)

    z = w_mat @ x_mat

    print(z)