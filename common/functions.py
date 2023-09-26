""" Some useful functions in various parts of BERNAISE. """
import dolfin as df
import ufl_legacy as ufl
__author__ = "Gaute Linga"


# Phase field chemical potential
def pf_potential(phi):
    """ Phase field potential. """
    return 0.25*(1.-phi**2)**2


def diff_pf_potential(phi):
    """ Derivative of the phase field potential. """
    return phi**3-phi


def diff_pf_potential_c(phi):
    """ Convex decomposition of the phase field potential. Positive part. """
    return phi**3


def diff_pf_potential_e(phi):
    """ Convex decomposition of the phase field potential. Negative part. """
    return phi


def diff_pf_potential_linearised(phi, phi0):
    """ Linearised derivative of phase field potential. """
    return phi0**3-phi0+(3*phi0**2-1.)*(phi-phi0)


def pf_contact(phi):
    """ Phase field contact function. """
    return (2. + 3.*phi - phi**3)/4.


def diff_pf_contact(phi):
    """ Derivative of phase field contact function. """
    return 3.*(1. - phi**2)/4.


def ddiff_pf_contact(phi):
    """ Double derivative of phase field contact. """
    return -3.*phi/2.


def diff_pf_contact_linearised(phi, phi0):
    """ Linearised derivative of phase field contact function. """
    return diff_pf_contact(phi0) + ddiff_pf_contact(phi0)*(phi-phi0)


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


def ramp_harmonic(phi, A):
    """ Weighted harmonic mean according to phi. """
    return (A[0]**(-1)*0.5*(1.+phi) + A[1]**(-1)*0.5*(1.-phi))**(-1)


def ramp_geometric(phi, A):
    """ Weighted geometric mean according to phi. """
    return A[0]**(0.5*(1.+phi))*A[1]**(0.5*(1.-phi))


# Filters
def dfabs(a):
    return abs(a)


def sign(a):
    return ufl.sign(a)


def max_value(a, b):
    return ufl.max_value(a, b)


def min_value(a, b):
    return ufl.min_value(a, b)


def unit_interval_filter(phi):
    return min_value(max_value(phi, -1.), 1.)


def absolute(q):
    return dfabs(q)


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
    c_min = min_value(c, c_cutoff)
    return (alpha_c(c_max) + alpha_cc(c_cutoff)*(c_min-c_cutoff))


def alpha_reg(c, c_cutoff):
    c_max = max_value(c, c_cutoff)
    c_min = min_value(c, c_cutoff)
    return (alpha(c_max) + alpha_c(c_cutoff)*(c_min-c_cutoff)
            + 0.5*alpha_cc(c_cutoff)*(c_min-c_cutoff)**2)
