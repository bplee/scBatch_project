"""
helper functions for various tasks
"""
import os

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
