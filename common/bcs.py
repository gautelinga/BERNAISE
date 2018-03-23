"""
This module defines a range of different boundary conditions.
"""
from dolfin import DirichletBC, Constant, Expression
import numpy as np

__author__ = "Gaute Linga"


class GenericBC:
    """ The general boundary conditon. Contains functions. """
    def __init__(self):
        pass

    def is_dbc(self):
        return False

    def is_nbc(self):
        return False

    def dbc(self, subspace, subdomains, mark):
        pass

    def nbc(self):
        pass


class Fixed(GenericBC):
    """ Fixed boundary conditon. """
    def __init__(self, value):
        if isinstance(value, Expression):
            self.value = value
        else:
            self.value = Constant(value)

    def is_dbc(self):
        return True

    def dbc(self, subspace, subdomains, mark):
        return DirichletBC(subspace, self.value, subdomains, mark)


class NoSlip(Fixed):
    def __init__(self):
        Fixed.__init__(self, (0., 0.))  # To be generalized for arbitrary dim.


class FreeSlip(GenericBC):
    # Class for implementing free slip in a certain direction
    def __init__(self, value, dim):
        if isinstance(value, Expression):
            self.value = value
        else:
            self.value = Constant(value)
        self.dim = dim

    def is_dbc(self):
        return True

    def dbc(self, subspace, subdomains, mark):
        return DirichletBC(subspace.sub(self.dim),
                           self.value, subdomains, mark)


class Slip(GenericBC):
    def __init__(self, value, dim):
        if isinstance(value, Expression):
            self.value = value
        else:
            self.value = Constant(value)
        self.dim = int(dim)

    def is_dbc(self):
        return True

    def dbc(self, subspace, subdomains, mark):
        return DirichletBC(subspace.sub(self.dim),
                           Constant(0.), subdomains, mark)

    def is_nbc(self):
        return True

    def nbc(self):
        return self.value


class Charged(GenericBC):
    def __init__(self, value):
        if isinstance(value, Expression):
            self.value = value
        else:
            self.value = Constant(value)

    def is_nbc(self):
        return True

    def nbc(self):
        return self.value


class ContactAngle(GenericBC):
    def __init__(self, theta):
        self.value = Constant(np.cos(theta))

    def is_nbc(self):
        return True

    def nbc(self):
        return self.value


class Pressure(Fixed):
    def is_nbc(self):
        return True

    def nbc(self):
        return self.value


class Open(GenericBC):
    def __init__(self, value):
        self.value = value

    def is_nbc(self):
        return True

    def nbc(self):
        return Constant(self.value)
