__author__ = "Gaute Linga"


__all__ = ["get_subproblems", "setup", "solve", "update"]


def get_subproblems(**namespace):
    """ Return dict of subproblems as defined by the solver. """
    return dict()


def setup(**namespace):
    """ Set up all equations that should be solved.
    Returns dict of solvers."""
    return dict()


def solve(**namespace):
    """ Solve at a timestep. """
    pass


def update(**namespace):
    """ Update work arrays at the end of timestep. """
    pass
