"""Visualizer"""


from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np


def image_visualizer(
    X: np.ndarray,
    Y: np.ndarray,
    classes: List[str],
    title: str,
    img_per_row: int = 7,
    img_per_col: int = 7,
    figsize: Tuple[float, float] = (15, 15),
    theme: str = "dark_background",
    adjust: Dict[str, float] = None,  # type: ignore
) -> None:
    """
    Visualize images with corresponding labels.

    Displays a grid of images along with their corresponding labels.

    Args:
        - X (np.ndarray): Array of input images with shape
        (num_images, img_height, img_width, num_channels).
        - Y (np.ndarray): Array of labels with shape (num_images,).
        - classes (List[str]): List of class labels corresponding to the labels in Y.
        - title (str): Title for the plot.
        - img_per_row (int, optional): Number of images to display per row. Defaults to 7.
        - img_per_col (int, optional): Number of images to display per column. Defaults to 7.
        - figsize (Tuple[float, float], optional): Figure size of the plot. Defaults to (15, 15).
        - theme (str, optional): Matplotlib style theme to use. Defaults to "dark_background".
        - adjust (Dict, optional): Parameters for adjusting the subplot layout. 
        Defaults to predefined values.

    Returns:
        None
    """
    if adjust is None:
        adjust = {
            "left": 0,
            "bottom": 0.029,
            "right": 1,
            "top": 0.916,
            "wspace": 0,
            "hspace": 0.31,
        }

    plt.style.use(theme)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    for index in range(img_per_row * img_per_col):
        plt.subplot(img_per_row, img_per_col, index + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[index])
        plt.xlabel(f"{classes[Y[index]]}")

    plt.subplots_adjust(
        left=adjust["left"],
        bottom=adjust["bottom"],
        right=adjust["right"],
        top=adjust["top"],
        wspace=adjust["wspace"],
        hspace=adjust["hspace"],
    )
    mng = plt.get_current_fig_manager()
    mng.set_window_title(title)
    plt.show(block=False)

    return None
