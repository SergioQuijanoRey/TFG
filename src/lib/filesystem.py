"""Operations related to file system, ie. saving / loading models from disk"""

import os
import pathlib
import torch
from torch import nn

from typing import Callable

# TODO -- I have seen that along the net we can save some state in form of a python dict
#      -- Source: https://www.programcreek.com/python/?code=drimpossible%2FDeep-Expander-Networks%2FDeep-Expander-Networks-master%2Fcode%2Fmodels%2F__init__.py
def save_model(net: nn.Module, folder_path: str, file_name: str) -> None:
    """
    Saves a model in memory

    Parameters:
    ===========
    net: the model we are saving to memory
    folder_path: the folder in which we want to save the model. If it does not exist, we create it
    file_name: the name of the file created in the folder
    """

    create_dir_if_not_exists(folder_path)
    save_path = os.path.join(folder_path, file_name)
    torch.save(net.state_dict(), save_path)

def load_model(model_path: str, model_class: Callable[nn.Module]) -> nn.Module:
    """
    Loads a model from disk and returns it

    Parameters:
    ===========
    model_path: the path to the disk file
    model_class: lambda function of the constructor of the net class
                 this way, we instance a new object of that class, and load the state from disk

    Returns:
    ========
    net: the loaded model
    """
    net = model_class()
    net.load_state_dict(torch.load(model_path))
    return net


def create_dir_if_not_exists(path: str) -> None:
    """Creates a dir if it does not exist"""

    if os.path.isdir(path) is False:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
