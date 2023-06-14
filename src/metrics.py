"""metrics"""
from typing import Any
import numpy as np


def binary_accuracy(y_prediction: np.ndarray, y_label: np.ndarray) -> np.floating[Any]:
    """Compute the accuracy of a prediction compared to the true labels.

    The accuracy is calculated as the fraction of correctly predicted labels
    out of the total number of labels.

    Args:
        y_prediction (np.ndarray): Predicted labels.
        y_label (np.ndarray): True labels.

    Returns:
        float: Accuracy as a value between 0 and 1, where 1 represents
        perfect accuracy.

    Notes:
        - The input arrays must have the same shape.
        - The labels are assumed to be binary (0 or 1).

    """
    return 1 - np.mean(np.abs(y_prediction - y_label))
