import os
import sys
from inspect import getsource


def get_valid_diva_models(dir="./"):
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
    files = os.listdir(dir)
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