"""Activations"""

from typing import Any
import numpy as np


def sigmoid(z: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Apply the sigmoid activation function element-wise to the input.

    The sigmoid function is a commonly used activation function in neural
    networks.
    It maps the input values to the range between 0 and 1.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the sigmoid function
        element-wise.

    Notes:
        - The sigmoid function is defined as sigmoid(z) = 1 / (1 + exp(-z)).
        - The input array can have any shape.

    References:
        - https://en.wikipedia.org/wiki/Sigmoid_function

    """
    return 1 / (1 + np.exp(-z))


def tanh(z: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Compute the hyperbolic tangent (tanh) function element-wise on the input array.

    The hyperbolic tangent function is defined as (exp(z) - exp(-z)) / (exp(z) + exp(-z)).
    It is a commonly used activation function in neural networks.

    Args:
        - z (np.ndarray): Input array.

    Returns:
        - np.ndarray: Output array after applying the tanh function element-wise.

    Raises:
        - TypeError: If the input array `z` is not of type `np.ndarray`.
        - ValueError: If the input array `z` is not numeric or does not have a floating-point
        data type.

    Examples:
        >>> z = np.array([-1.0, 0.0, 1.0])
        >>> tanh(z)
        array([-0.76159416,  0.        ,  0.76159416])
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def relu(z):
    """
    Compute the Rectified Linear Unit (ReLU) function element-wise on the input array.

    The ReLU function is defined as max(0, z). It returns 0 for negative values of z
    and leaves positive values unchanged.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the ReLU function element-wise.

    Raises:
        TypeError: If the input array `z` is not of type `np.ndarray`.
        ValueError: If the input array `z` is not numeric or does not have a floating-point data type.

    Examples:
        >>> z = np.array([-1.0, 0.0, 1.0])
        >>> relu(z)
        array([0., 0., 1.])
    """
    return np.maximum(0.0, z)

def leakyRelu(z, factor=0.01):
    """_summary_

    Args:
        z (_type_): _description_
        factor (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    return np.maximum(factor * z, z)


def dsigmoid(z):
    """_summary_

    Args:
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


def dtanh(z):
    """_summary_

    Args:
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 - np.power(tanh(z), 2)


def drelu(z):
    """_summary_

    Args:
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 * (z >= 0)


def dleakyRelu(z, factor=0.01):
    """_summary_

    Args:
        z (_type_): _description_
        factor (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    return (z < 0) * factor + (z >= 0) * 1
