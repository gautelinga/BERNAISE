"""
This module defines a range of different boundary conditions.
"""
from dolfin import DirichletBC, Constant

__author__ = "Gaute Linga"


class GenericBC:
    """ The general boundary conditon. Contains functions. """
    def __init__(self):
        pass


class Fixed(GenericBC):
    """ Fixed boundary conditon. """
    def __init__(self, value):
        self.value = value

    def dbc(self, subspace, subdomains, mark):
        return DirichletBC(subspace, Constant(self.value), subdomains, mark)


class NoSlip(Fixed):
    def __init__(self):
        Fixed.__init__((0., 0.))  # To be generalized for arbitrary dim.
