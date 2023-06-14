"""Initializers"""


from typing import Tuple, Dict, List
import numpy as np


def initialize_with_zeros(dim: int) -> Tuple[np.ndarray, float]:
    """Initialize weight vector and bias term with zeros.

    Args:
        dim (int): Number of features in the weight vector.

    Returns:
        Tuple[np.ndarray, float]: Initial weight vector and bias term.

    Notes:
        - The weight vector is a column vector of shape (dim, 1).
        - The bias term is initialized as 0.0.
    """
    return np.zeros(dim).reshape(dim, 1), 0.0


def random_initializer(layers: List, factor: float = 0.1) -> Dict[str, np.ndarray]:
    """Random Initializer"""
    parameters: Dict = {}
    hidden_layers = layers[1:]
    for i, _ in enumerate(hidden_layers):
        # print(f"W{i+1} = ({hidden_layers[i]}, {layers[i]})")
        # print(f"b{i+1} = ({hidden_layers[i]}, 1)\n")

        parameters[f"W{i+1}"] = np.random.randn(hidden_layers[i], layers[i]) * factor
        parameters[f"b{i+1}"] = np.zeros((hidden_layers[i], 1))

    return parameters
