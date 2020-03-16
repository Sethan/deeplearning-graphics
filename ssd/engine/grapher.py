import numpy as np
import matplotlib.pyplot as plt



def plot_loss(loss_dict: dict, label: str = None, fmt="-"):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
    """
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    plt.plot(global_steps, loss, fmt, label=label)
