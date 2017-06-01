""" Some useful functions in various parts of BERNAISE. """
import dolfin as df
__author__ = "Gaute Linga"


# Phase field chemical potential
def pf_potential(phi):
    """ Phase field potential. """
    return 0.25*(1.-phi**2)**2


def diff_pf_potential(phi):
    """ Derivative pf the phase field potential. """
    return phi**3-phi


def diff_pf_potential_linearised(phi, phi0):
    """ Linearised derivative of phase field potential. """
    return phi0**3-phi0+(3*phi0**2-1.)*(phi-phi0)


# Phase field auxiliary fields
def ramp(phi, A):
    """
    Ramps linearly between A[0] and A[1] according to phi,
    such that phi=1 => A(phi) = A[1], phi=-1 => A(phi) = A[2]
    """
    return A[0]*0.5*(1.+phi) + A[1]*0.5*(1.-phi)


def dramp(A):
    """ Derivative of ramping function above. Returns df.Constant."""
    return df.Constant(0.5*(A[0]-A[1]))
