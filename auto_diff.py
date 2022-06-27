# https://python-course.eu/oop/magic-methods.php
# https://sidsite.com/posts/autodiff/

from collections import defaultdict

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

def add(a, b):
    "Create the variable that results from adding two variables."
    value = a.value + b.value
    local_gradients = (
        (a, 1), # local derivative with respect to a is 1
        (b, 1) # local derivative with respect to b is 1
    )
    return Variable(value, local_gradients)

def mul(a, b):
    "Create the variable that results from multiplying two variables."
    value = a.value * b.value
    local_gradients = (
        (a, b.value),
        (b, a.value)
    )
    return Variable(value, local_gradients)

def get_gradients(variable):
    gradients = defaultdict(lambda: 0)

    def compute_gradients(variable, path_value):
        for child_variable, local_gradient in variable.local_gradients:
            # Multiply the edges of a path
            value_of_path_to_child = path_value * local_gradient
            
            # Add together the different paths
            gradients[child_variable] += value_of_path_to_child

            # Recurse through graph
            compute_gradients(child_variable, value_of_path_to_child)

    compute_gradients(variable, path_value=1)

    return gradients

a = Variable(4)
b = Variable(3)
c = Variable(10)

w = a + b + c
grad = get_gradients(w)

print(w.value)