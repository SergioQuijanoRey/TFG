"""
Module to do specific operations with tensorboard
"""

import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def get_writer(name: str = None):
    """
    Returns the writer to tensorboard that we are using

    Currently using default `log_dir` which is "runs"

    Parameters:
    ===========
    name: the name of the current run.
          Default name used when None passed as parameter
          Default value specified by tensorboard
    """

    # No name given
    if name is None:
        return SummaryWriter()

    # Get full path and use it
    log_dir = os.path.join("runs", name)
    return SummaryWriter(log_dir = log_dir)

# BUG -- is not working
def net_architecture_to_tensorboard(net: nn.Module):
    """
    Sends a net to tensorboard, to display the graph of the net

    Parameters:
    ============
    net: the nn we want to display
    """
    writer = get_writer()
    writer.add_graph(net)
    writer.close()
