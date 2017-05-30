__author__ = "Gaute"


__all__ = ["setup", "get_solvers", "solve", "update"]


def setup(**namespace):
    """ Set up all equations that should be solved.
    Returns dict of ..."""
    return {}


def get_solvers(**namespace):
    """ Return the linear solvers required in the problem. """
    pass


def solve(**namespace):
    """ Solve at a timestep. """
    pass


def update(**namespace):
    """ Update work arrays at the end of timestep. """
    pass
