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
    such that phi=1 => A(phi) = A[0], phi=-1 => A(phi) = A[1]
    """
    return A[0]*0.5*(1.+phi) + A[1]*0.5*(1.-phi)


def dramp(A):
    """ Derivative of ramping function above. Returns df.Constant."""
    return df.Constant(0.5*(A[0]-A[1]))


# Filters
def max_value(a, b):
    return 0.5*(a+b+df.sign(a-b)*(a-b))


def min_value(a, b):
    return 0.5*(a+b-df.sign(a-b)*(a-b))


def unit_interval_filter(phi):
    return min_value(max_value(phi, -1.), 1.)


def absolute(q):
    return df.sign(q)*q


# Chemical potential functions
def alpha(c):
    return c*(df.ln(c)-1)


def alpha_c(c):
    return df.ln(c)


def alpha_cc(c):
    return 1./c


# Regulated chemical potential
def alpha_cc_reg(c, c_cutoff):
    return alpha_cc(max_value(c, c_cutoff))


def alpha_c_reg(c, c_cutoff):
    c_max = max_value(c, c_cutoff)
    dc = c_max - c_cutoff
    return (alpha_c(c_max) - alpha_c(c_cutoff)
            + alpha_cc(c_cutoff)*c
            - alpha_cc(c_cutoff)*dc)


def alpha_reg(c, c_cutoff):
    c_max = max_value(c, c_cutoff)
    dc = c_max-c_cutoff
    return (alpha(c_max) - alpha(c_cutoff)
            + 0.5*alpha_cc(c_cutoff)*c**2
            - alpha_c(c_cutoff)*dc
            - 0.5*alpha_cc(c_cutoff)*dc**2, c_cutoff)
