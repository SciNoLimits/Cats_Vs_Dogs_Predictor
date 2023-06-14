"""Propagations"""

import numpy as np
from activations import relu, sigmoid, drelu


def forward_propagation(X, parameters, layers):
    """_summary_

    Args:
        X (_type_): _description_
        parameters (_type_): _description_
        layers (_type_): _description_

    Returns:
        _type_: _description_
    """
    cache = {"A0": X}
    for i in range(len(layers) - 1):  # type: ignore
        if i + 1 != len(layers) - 1:
            # print(f"Z{i+1} = W{i+1} * A{i} + b{i+1}")
            # print(f"A{i+1} = relu(Z{i+1})\n")

            cache[f"Z{i+1}"] = (
                np.dot(parameters[f"W{i+1}"], cache[f"A{i}"]) + parameters[f"b{i+1}"]
            )
            cache[f"A{i+1}"] = relu(cache[f"Z{i+1}"])
        else:
            # print(f"Z{i+1} = W{i+1} * A{i} + b{i+1}")
            # print(f"A{i+1} = sigmoid(Z{i+1})\n")

            cache[f"Z{i+1}"] = (
                np.dot(parameters[f"W{i+1}"], cache[f"A{i}"]) + parameters[f"b{i+1}"]
            )
            cache[f"A{i+1}"] = sigmoid(cache[f"Z{i+1}"])

    AL = cache[f"A{len(layers)-1}"]

    return AL, cache


def backward_propagation(X, Y, parameters, cache, layers):
    """_summary_

    Args:
        X (_type_): _description_
        Y (_type_): _description_
        parameters (_type_): _description_
        cache (_type_): _description_
        layers (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = X.shape[1]
    grads = {}
    for i in range(len(layers) - 1, 0, -1):
        if i == len(layers) - 1:
            # print(f"dZ{i} = A{i} - Y")
            # print(f"dW{i} = dZ{i} * A{i-1} /m")
            # print(f"db{i} = np.sum(dZ{i}, axis=1, keepdims=True)/m \n")
            grads[f"dZ{i}"] = cache[f"A{i}"] - Y
        else:
            # print(f"dZ{i} = (W{i+1}.T . dZ{i+1}) * drelu(A{i})")
            # print(f"dW{i} = dZ{i} * A{i-1} /m")
            # print(f"db{i} = np.sum(dZ{i} axis=1, keepdims=True)/m \n")

            grads[f"dZ{i}"] = np.multiply(
                np.dot(parameters[f"W{i+1}"].T, grads[f"dZ{i+1}"]),
                drelu(cache[f"A{i}"]),
            )

        grads[f"dW{i}"] = (np.dot(grads[f"dZ{i}"], cache[f"A{i-1}"].T)) / m
        grads[f"db{i}"] = np.sum(grads[f"dZ{i}"], axis=1, keepdims=True) / m

    return grads
