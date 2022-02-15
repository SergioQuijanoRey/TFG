"""Operations related to file system, ie. saving / loading models from disk"""

import os
import pathlib
import torch
from torch import nn

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

# TODO -- change model_class to a lambda that initializes a net instance
#   This way, we can initialize classes with parameters (such building block of ResNet)
def load_model(model_path: str, model_class: nn.Module) -> nn.Module:
    """
    Loads a model from disk and returns it

    Parameters:
    ===========
    model_path: the path to the disk file
    model_class: the class corresponding to the model in disk
                 because we need to create an object of some given type

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
