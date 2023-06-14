"""Losses"""

from typing import Any
import numpy as np


def binary_cross_entropy_loss(
    activation: np.ndarray[Any, Any], y_label: np.ndarray[Any, Any]
) -> Any:
    """Computes the binary cross-entropy loss between predicted activations and true labels.

    This function calculates the binary cross-entropy loss by comparing the predicted
    activations from the model with the true labels.

    Args:
        activation (np.ndarray): Predicted activations from the model.
            Shape: (1, n_samples).

        y_label (np.ndarray): True labels.
            Shape: (1, n_samples).

    Returns:
        Any: Binary cross-entropy loss value.

    Notes:
        - The binary cross-entropy loss is calculated as the sum of two terms:
        one for the positive class (y=1) and one for the negative class (y=0).
        - The loss is averaged over the number of samples (n_samples).

    Warning:
        - The input arrays (activation and y_label) should have the same shape.

    References:
        - https://en.wikipedia.org/wiki/Cross_entropy

    """
    return np.mean(
        -(y_label * np.log(activation) + (1 - y_label) * np.log(1 - activation))
    )
