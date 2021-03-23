import os
import sys
from inspect import getsource


# diva specific commands

def get_valid_diva_models(folder="./"):
    """
    Collects the names of DIVA models in `dir`
    These are NOT filepaths or file names.

    Parameters
    ----------
    dir : str
        String pertaining to valid file path

    Returns
    -------
    lst
        list of model names without file extensions,
        for which there exist "<name>.model" and "<name>.config" within `dir`

    """
    files = os.listdir(folder)
    model_file_exts = set(("model", "config"))
    name_exts = [f.split(".") for f in files if f.split(".")[-1] in model_file_exts]
    model_names = []
    while name_exts:
        curr = name_exts.pop()
        for i, f in enumerate(name_exts):
            # if the model names are the same
            if f[0] == curr[0]:
                # if there is both the ".model" and ".config" extension
                if set((f[-1], curr[-1])) == model_file_exts:
                    model_names.append(curr[0])
    return model_names


# logistical commands

def ensure_dir(filepath):
    """
    Makes sure directory at `filepath` exists, if not, then it creates it

    Parameters
    ----------
    filepath : str
        filepath to a directory that you want to check

    Returns
    -------
    None
        creates directory

    """
    if not os.path.exists(filepath):
        print(f'Directory {filepath} does not exist. Creating directory in {os.getcwd()}.')
        os.makedirs(filepath)

def wrap(a):
    """

    Parameters
    ----------
    a : something

    Returns
    -------
    list
        wraps a if it isn't a list already
    """

    rtn = a if type(a) == list else [a]
    return rtn