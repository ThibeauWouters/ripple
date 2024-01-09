"""Small utility script for shared functions between tidal waveforms, especially for NRTidalv2"""

import jax
import jax.numpy as jnp
from ..typing import Array
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, TWO_PI, MSUN, MRSUN, GMsun_over_c2_Gpc, GMsun_over_c3
from .IMRPhenomD_utils import get_fRD_fdamp
# ### DEBUG
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
# ### DEBUG

def universal_relation(coeffs: Array, x: float):
    """Applies the general formula of a universal relationship, which is a quartic polynomial.

    Args:
        coeffs (Array): Array of coefficients for the quartic polynomial, starting from the constant term and going to the fourth order.
        x (float): Variable of quartic polynomial

    Returns:
        float: Result of universal relation
    """
    return coeffs[0] + coeffs[1] * x + coeffs[2] * (x ** 2) + coeffs[3] * (x ** 3) + coeffs[4] * (x ** 4)


def get_quadparam_octparam(lambda_: float) -> tuple[float, float]:
    """Compute the quadrupole and octupole parameter by checking the value of lambda and choosing the right subroutine.
    If lambda is smaller than 1, we make use of the fit formula as given by the LAL source code. Otherwise, we rely on the equations of
    the NRTidalv2 paper to get these parameters.

    Args:
        lambda_ (float): Tidal deformability of object.

    Returns:
        tuple[float, float]: Quadrupole and octupole parameters.
    """
    
    # # Avoid NaNs: make sure the lambda used here is never negative (https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where)
    # lambda_ = jnp.where(lambda_ < 1e-10, 1e-10, lambda_)
    
    # jax.debug.print("lambda in get_quadparam_octparam:")
    # jax.debug.print(str(lambda_))
    
    # jax.debug.print("shape lambda in get_quadparam_octparam:")
    # shape = jnp.shape(lambda_)
    # jax.debug.print(str(shape))
    
    # jax.debug.print("negative lambda in get_quadparam_octparam:")
    # shape = jnp.shape(lambda_)
    # is_negative = jnp.any(lambda_ < 0.)
    # jax.debug.print(str(is_negative))
    # jax.debug.print("is negative lambda: sum")
    # is_negative_sum = jnp.sum(is_negative).astype(float)
    # jax.debug.print(str(is_negative_sum))
    
    # jax.debug.print("iterating over")
    # if not isinstance(is_negative_sum, float):
    #     for element in is_negative:
    #         jax.debug.print(str(element))

    # TODO this is gwfast implementation, make sure is OK and refactor more nicely
    quadparam = jnp.where(lambda_ < 1., 
                          1. + lambda_*(0.427688866723244 + lambda_*(-0.324336526985068 + lambda_*0.1107439432180572)), 
                          jnp.exp(0.1940 + 0.09163 * jnp.log(lambda_) + 0.04812 * jnp.log(lambda_) * jnp.log(lambda_) -4.283e-3 * jnp.log(lambda_) * jnp.log(lambda_) * jnp.log(lambda_) + 1.245e-4 * jnp.log(lambda_) * jnp.log(lambda_) * jnp.log(lambda_) * jnp.log(lambda_)))
    
    octparam = jnp.exp(0.003131 + 2.071 * jnp.log(quadparam)  - 0.7152 * jnp.log(quadparam) * jnp.log(quadparam) + 0.2458 * jnp.log(quadparam) * jnp.log(quadparam) * jnp.log(quadparam) - 0.03309 * jnp.log(quadparam) * jnp.log(quadparam) * jnp.log(quadparam) * jnp.log(quadparam))
    
    return quadparam, octparam
    
    
# def _get_quadparam_octparam_low(lambda_: float) -> tuple[float, float]:
#     """
#     Computes quadparameter, see eq (28) of NRTidalv2 paper and also LALSimUniversalRelations.c of lalsuite
    
#     Version for lambdas smaller than 1.
    
#     LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
#     Args:
#         lambda_: tidal deformability

#     Returns:
#         quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
#         octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
#     """
    
#     # Coefficients of universal relation
#     oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]
    
#     # Extension of the fit in the range lambda2 = [0,1.] so that the BH limit is enforced, lambda2bar->0 gives quadparam->1. and the junction with the universal relation is smooth, of class C2
#     quadparam = 1. + lambda_ * (0.427688866723244 + lambda_ * (-0.324336526985068 + lambda_ * 0.1107439432180572))
#     log_quadparam = jnp.log(quadparam)
        
#     # Compute octparam:
#     log_octparam = universal_relation(oct_coeffs, log_quadparam)
#     octparam = jnp.exp(log_octparam)

#     return quadparam, octparam

# def _get_quadparam_octparam_high(lambda_: float) -> tuple[float, float]:
#     """
#     Computes quadparameter, see eq (28) of NRTidalv2 paper and also LALSimUniversalRelations.c of lalsuite
    
#     Version for lambdas greater than 1.
    
#     LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
#     Args:
#         lambda_: tidal deformability

#     Returns:
#         quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
#         octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
#     """
    
#     # Coefficients of universal relation
#     quad_coeffs = [0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4]
#     oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]
        
#     # High lambda (above 1): use universal relation
#     jax.debug.print("lambda in high fn:")
#     jax.debug.print(str(lambda_))
#     log_lambda = jnp.log(lambda_)
#     jax.debug.print("log_lambda")
#     jax.debug.print(str(log_lambda))
#     log_quadparam = universal_relation(quad_coeffs, log_lambda)
    
#     # Compute octparam:
#     log_octparam = universal_relation(oct_coeffs, log_quadparam)

#     quadparam = jnp.exp(log_quadparam)
#     octparam = jnp.exp(log_octparam)

#     return quadparam, octparam

def get_kappa(theta: Array) -> float:
    """Computes the tidal deformability parameter kappa according to equation (8) of the NRTidalv2 paper.

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
    term1 = (1.0 + 12.0 * X2 / X1) * (X1 ** 5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2 ** 5.0) * lambda2
    kappa = (3./13.) * (term1 + term2)
    
    return kappa 

def get_amp0_lal(M: float, distance: float):
    """Get the amp0 prefactor as defined in LAL in LALSimIMRPhenomD, line 331. 

    Args:
        M (float): Total mass in solar masses
        distance (float): Distance to the source in Mpc.

    Returns:
        float: amp0 from LAL.
    """
    amp0 = 2. * jnp.sqrt(5. / (64. * PI)) * M * MRSUN * M * gt / distance
    return amp0


def _get_spin_induced_quadrupole_phase_coeff(lambda_: float, mass: float) -> float:
    """Compute the quantity from equation (11) from http://arxiv.org/abs/1503.05405

    Args:
        lambda_ (float): Tidal deformability of object
        mass (float): Mass of object in solar masses

    Returns:
        float: a(m) as defined in equation (11) of http://arxiv.org/abs/1503.05405
    """
    
    # Auxiliary parameter:
    # TODO what if lambda is zero or negative?
    x = jnp.log(lambda_ / (mass ** 5))
    coeffs = jnp.array([0.194, 0.0936, 0.0474, -4.21e-3, 1.23e-4])
    
    ln_a = universal_relation(coeffs, x)
    a = jnp.exp(ln_a)
        
    return a

def get_spin_induced_quadrupole_phase(v: Array, theta: Array) -> Array:
    """Computes the contribution to the phase from the spin-induced quadrupole moment

    Args:
        x (Array): Array of velocities, that is, v = (pi M f)^(1/3)
        theta (Array): Array of parameters (m1, m2, chi1, chi2, lambda1, lambda2)   

    Returns:
        Array: Phase contribution from spin-induced quadrupole moment
    """
    # See http://arxiv.org/abs/1503.05405, around eq (11)
    
    # Get parameters and auxiliary variables
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    M  = m1 + m2
    X1 = m1 / M
    X2 = m2 / M
    eta = m1 * m2 / (M ** 2.0)
    
    # Compute the spin-induced quadrupole phase (NOTE this is assuming aligned spin)
    a1 = _get_spin_induced_quadrupole_phase_coeff(lambda1, m1)
    a2 = _get_spin_induced_quadrupole_phase_coeff(lambda2, m2)
    sigma_qm_1 = 5 * a1 * (X1 ** 2) * chi1 ** 2
    sigma_qm_2 = 5 * a2 * (X2 ** 2) * chi2 ** 2
    
    sigma_qm = sigma_qm_1 + sigma_qm_2
    psi_qm = - (30 / (128 * eta)) * sigma_qm / v
    
    return psi_qm