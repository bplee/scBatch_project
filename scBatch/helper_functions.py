"""
helper functions for various tasks
"""


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