"""
Small utility script for shared functions between tidal waveforms, especially for NRTidalv2
"""

import jax
import jax.numpy as jnp
from ..typing import Array
from ripple.constants import gt, PI, MRSUN


def universal_relation(coeffs: Array, x: float):
    """
    Applies the general formula of a universal relationship, which is a quartic polynomial.

    Args:
        coeffs (Array): Array of coefficients for the quartic polynomial, starting from the constant term and going to the fourth order.
        x (float): Variable of quartic polynomial

    Returns:
        float: Result of universal relation
    """
    return (
        coeffs[0]
        + coeffs[1] * x
        + coeffs[2] * (x**2)
        + coeffs[3] * (x**3)
        + coeffs[4] * (x**4)
    )


def get_quadparam_octparam(lambda_: float) -> tuple[float, float]:
    """
    Compute the quadrupole and octupole parameter by checking the value of lambda and choosing the right subroutine.
    If lambda is smaller than 1, we make use of the fit formula as given by the LAL source code. Otherwise, we rely on the equations of
    the NRTidalv2 paper to get these parameters.

    Args:
        lambda_ (float): Tidal deformability of object.

    Returns:
        tuple[float, float]: Quadrupole and octupole parameters.
    """

    # Check if lambda is low or not, and choose right subroutine
    is_low_lambda = lambda_ < 1
    return jax.lax.cond(
        is_low_lambda,
        _get_quadparam_octparam_low,
        _get_quadparam_octparam_high,
        lambda_,
    )


def _get_quadparam_octparam_low(lambda_: float) -> tuple[float, float]:
    """
    Computes quadparameter following LALSimUniversalRelations.c of lalsuite

    Version for lambdas smaller than 1.

    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """

    # Coefficients of universal relation
    oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]

    # Extension of the fit in the range lambda2 = [0,1.] so that the BH limit is enforced, lambda2bar->0 gives quadparam->1. and the junction with the universal relation is smooth, of class C2
    quadparam = 1.0 + lambda_ * (
        0.427688866723244
        + lambda_ * (-0.324336526985068 + lambda_ * 0.1107439432180572)
    )
    log_quadparam = jnp.log(quadparam)

    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam


def _get_quadparam_octparam_high(lambda_: float) -> tuple[float, float]:
    """
    Computes quadparameter, following LALSimUniversalRelations.c of lalsuite

    Version for lambdas greater than 1.

    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """

    # Coefficients of universal relation
    quad_coeffs = [0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4]
    oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]

    # High lambda (above 1): use universal relation
    log_lambda = jnp.log(lambda_)
    log_quadparam = universal_relation(quad_coeffs, log_lambda)

    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)

    quadparam = jnp.exp(log_quadparam)
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam


def get_kappa(theta: Array) -> float:
    """
    Computes the tidal deformability parameter kappa according to equation (8) of the NRTidalv2 paper.

    Args:
        theta (Array): Intrinsic parameters m1, m2, chi1, chi2, lambda1, lambda2

    Returns:
        float: kappa_eff^T from equation (8) of NRTidalv2 paper.
    """

    # Auxiliary variables
    m1, m2, _, _, lambda1, lambda2 = theta
    M = m1 + m2
    X1 = m1 / M
    X2 = m2 / M

    # Get kappa
    term1 = (1.0 + 12.0 * X2 / X1) * (X1**5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2**5.0) * lambda2
    kappa = (3.0 / 13.0) * (term1 + term2)

    return kappa


def get_tidal_phase(x: Array, theta: Array, kappa: float = None) -> Array:
    """
    Computes the tidal phase psi_T from equation (17) of the NRTidalv2 paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters in the order (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (float): Tidal parameter kappa, precomputed in the main function.

    Returns:
        Array: Tidal phase correction.
    """

    # Compute auxiliary quantities
    m1, m2, _, _, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s

    X1 = m1_s / M_s
    X2 = m2_s / M_s
    
    if kappa is None:
        kappa = get_kappa(theta)

    # Compute powers
    x_2 = x ** (2.0)
    x_3 = x ** (3.0)
    x_3over2 = x ** (3.0 / 2.0)
    x_5over2 = x ** (5.0 / 2.0)

    # Initialize the coefficients
    c_Newt = 2.4375
    n_1 = -12.615214237993088
    n_3over2 = 19.0537346970349
    n_2 = -21.166863146081035
    n_5over2 = 90.55082156324926
    n_3 = -60.25357801943598
    d_1 = -15.111207827736678
    d_3over2 = 22.195327350624694
    d_2 = 8.064109635305156

    # Pade approximant
    num = (
        1.0
        + (n_1 * x)
        + (n_3over2 * x_3over2)
        + (n_2 * x_2)
        + (n_5over2 * x_5over2)
        + (n_3 * x_3)
    )
    den = 1.0 + (d_1 * x) + (d_3over2 * x_3over2) + (d_2 * x_2)
    ratio = num / den

    # Assemble everything
    psi_T = -kappa * c_Newt / (X1 * X2) * x_5over2
    psi_T *= ratio

    return psi_T

# TODO: might have to move this to a different file
def get_amp0_lal(M: float, distance: float):
    """
    Get the amp0 prefactor as defined in LAL in LALSimIMRPhenomD, line 331.

    Args:
        M (float): Total mass in solar masses
        distance (float): Distance to the source in Mpc.

    Returns:
        float: amp0 from LAL.
    """
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M * MRSUN * M * gt / distance
    return amp0