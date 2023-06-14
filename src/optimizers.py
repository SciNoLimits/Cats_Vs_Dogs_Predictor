"""Update Parameters"""

import copy

def update_parameters(parameters, grads, layers, learning_rate=0.001):
    parameters = copy.deepcopy(parameters)

    for i in range(len(layers) - 1):
        # print(f"W{i+1} -= learning_rate * dW{i+1}")
        # print(f"b{i+1} -= learning_rate * db{i+1} \n")
        parameters[f"W{i+1}"] -= learning_rate * grads[f"dW{i+1}"]
        parameters[f"b{i+1}"] -= learning_rate * grads[f"db{i+1}"]

    return parameters