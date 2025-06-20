import jax
import jax.numpy as jnp
from ..constants import PI, gt
from ..typing import Array


def get_cutoff_fMs(m1: float, m2: float, chi1: float, chi2: float):
    """
    Calculate characteristic frequencies in units of fM_s.
    Following LALSimIMRPhenomX_internals.c
    """
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    # QNM frequency fit
    # Simplified version of the full LALSuite fit
    a_final = jnp.clip(chi1 + chi2, -0.998, 0.998)  # Final spin estimate
    
    # Ring-down frequency (Eq. 5.6 in gr-qc/0512169)
    fMs_RD = (1.5251 - 1.1568 * (1.0 - a_final)**0.1292) / (2.0 * PI)
    
    # Damping frequency
    fMs_damp = (0.7000 + 1.4187 * (1.0 - a_final)**(-0.4990)) / (2.0 * PI)
    
    # MECO frequency (most efficient circular orbit)
    fMs_MECO = 1.0 / (6.0**(3.0/2.0) * PI)
    
    # ISCO frequency 
    fMs_ISCO = 1.0 / (6.0**(3.0/2.0) * PI)  # Simplified, should depend on spin
    
    return fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO


def wigner_d_matrix_element(l: int, m: int, mp: int, beta: Array) -> Array:
    """
    Calculate Wigner d-matrix elements d^l_{m,m'}(β) using JAX.
    Optimized for the specific cases needed in IMRPhenomXPHM.
    
    Args:
        l: Angular momentum quantum number
        m: Azimuthal quantum number  
        mp: Azimuthal quantum number (m')
        beta: Euler angle β
        
    Returns:
        d^l_{m,m'}(β)
    """
    cos_beta = jnp.cos(beta)
    sin_beta = jnp.sin(beta)
    cos_half_beta = jnp.cos(beta / 2.0)
    sin_half_beta = jnp.sin(beta / 2.0)
    
    # For l=2 modes (most important)
    if l == 2:
        if m == 2 and mp == 2:
            return cos_half_beta**4
        elif m == 2 and mp == 1:
            return 2.0 * cos_half_beta**3 * sin_half_beta
        elif m == 2 and mp == 0:
            return jnp.sqrt(6.0) * cos_half_beta**2 * sin_half_beta**2
        elif m == 2 and mp == -1:
            return 2.0 * cos_half_beta * sin_half_beta**3
        elif m == 2 and mp == -2:
            return sin_half_beta**4
        elif m == 1 and mp == 2:
            return -2.0 * cos_half_beta**3 * sin_half_beta
        elif m == 1 and mp == 1:
            return cos_half_beta**2 * (2.0 * cos_beta + 1.0)
        elif m == 1 and mp == 0:
            return jnp.sqrt(6.0) * cos_half_beta * sin_half_beta * cos_beta
        elif m == 1 and mp == -1:
            return sin_half_beta**2 * (2.0 * cos_beta - 1.0)
        elif m == 1 and mp == -2:
            return 2.0 * cos_half_beta * sin_half_beta**3
        elif m == 0 and mp == 2:
            return jnp.sqrt(6.0) * cos_half_beta**2 * sin_half_beta**2
        elif m == 0 and mp == 1:
            return -jnp.sqrt(6.0) * cos_half_beta * sin_half_beta * cos_beta
        elif m == 0 and mp == 0:
            return 0.5 * (3.0 * cos_beta**2 - 1.0)
        elif m == 0 and mp == -1:
            return jnp.sqrt(6.0) * cos_half_beta * sin_half_beta * cos_beta
        elif m == 0 and mp == -2:
            return jnp.sqrt(6.0) * cos_half_beta**2 * sin_half_beta**2
        elif m == -1 and mp == 2:
            return -2.0 * cos_half_beta * sin_half_beta**3
        elif m == -1 and mp == 1:
            return sin_half_beta**2 * (2.0 * cos_beta - 1.0)
        elif m == -1 and mp == 0:
            return -jnp.sqrt(6.0) * cos_half_beta * sin_half_beta * cos_beta
        elif m == -1 and mp == -1:
            return cos_half_beta**2 * (2.0 * cos_beta + 1.0)
        elif m == -1 and mp == -2:
            return 2.0 * cos_half_beta**3 * sin_half_beta
        elif m == -2 and mp == 2:
            return sin_half_beta**4
        elif m == -2 and mp == 1:
            return -2.0 * cos_half_beta * sin_half_beta**3
        elif m == -2 and mp == 0:
            return jnp.sqrt(6.0) * cos_half_beta**2 * sin_half_beta**2
        elif m == -2 and mp == -1:
            return -2.0 * cos_half_beta**3 * sin_half_beta
        elif m == -2 and mp == -2:
            return cos_half_beta**4
    
    # For l=3 modes
    elif l == 3:
        # Implement key l=3 Wigner-d elements
        if m == 3 and mp == 3:
            return cos_half_beta**6
        elif m == 3 and mp == 2:
            return jnp.sqrt(3.0) * cos_half_beta**5 * sin_half_beta
        elif m == 3 and mp == 1:
            return jnp.sqrt(15.0) * cos_half_beta**4 * sin_half_beta**2
        elif m == 3 and mp == 0:
            return jnp.sqrt(10.0) * cos_half_beta**3 * sin_half_beta**3
        elif m == 3 and mp == -1:
            return jnp.sqrt(15.0) * cos_half_beta**2 * sin_half_beta**4
        elif m == 3 and mp == -2:
            return jnp.sqrt(3.0) * cos_half_beta * sin_half_beta**5
        elif m == 3 and mp == -3:
            return sin_half_beta**6
        # Add more l=3 cases as needed
        else:
            # Fallback to general formula for other cases
            return _wigner_d_general(l, m, mp, beta)
    
    # For l=4 modes  
    elif l == 4:
        if m == 4 and mp == 4:
            return cos_half_beta**8
        # Add more l=4 cases as needed
        else:
            return _wigner_d_general(l, m, mp, beta)
    
    else:
        return _wigner_d_general(l, m, mp, beta)


def _wigner_d_general(l: int, m: int, mp: int, beta: Array) -> Array:
    """
    General Wigner d-matrix calculation for cases not optimized above.
    Uses the series representation.
    """
    cos_half_beta = jnp.cos(beta / 2.0)
    sin_half_beta = jnp.sin(beta / 2.0)
    
    # Find the summation limits
    k_min = jnp.maximum(0, m - mp)
    k_max = jnp.minimum(l + m, l - mp)
    
    # Initialize sum
    result = 0.0
    
    # Use a fixed range and mask invalid terms
    for k in range(10):  # Should be sufficient for l <= 4
        valid_k = (k >= k_min) & (k <= k_max)
        
        # Calculate binomial coefficients
        binom1 = _binomial_coeff(l + m, k)
        binom2 = _binomial_coeff(l - m, k + mp - m)
        
        # Calculate the term
        sign = (-1)**(k + mp - m)
        cos_term = cos_half_beta**(2*l + m - mp - 2*k)
        sin_term = sin_half_beta**(2*k + mp - m)
        
        term = sign * binom1 * binom2 * cos_term * sin_term
        result += jnp.where(valid_k, term, 0.0)
    
    # Overall normalization
    norm = jnp.sqrt(_binomial_coeff(l + m, l + mp) * _binomial_coeff(l - m, l - mp))
    
    return norm * result


def _binomial_coeff(n: int, k: int) -> Array:
    """
    Binomial coefficient calculation using gamma functions for JAX compatibility.
    """
    n_f = jnp.float64(n + 1)
    k_f = jnp.float64(k + 1) 
    nk_f = jnp.float64(n - k + 1)
    return jnp.exp(jax.lax.lgamma(n_f) - jax.lax.lgamma(k_f) - jax.lax.lgamma(nk_f))


def spherical_harmonic_Ylm(l: int, m: int, theta: Array, phi: Array) -> Array:
    """
    Spherical harmonics Y_l^m(θ,φ) for the modes needed in IMRPhenomXPHM.
    
    Args:
        l: Degree
        m: Order  
        theta: Polar angle
        phi: Azimuthal angle
        
    Returns:
        Y_l^m(θ,φ)
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    
    # Associated Legendre polynomials for specific cases
    if l == 2:
        if m == 0:
            P_lm = 0.5 * (3.0 * cos_theta**2 - 1.0)
            norm = jnp.sqrt(5.0 / (4.0 * PI))
        elif abs(m) == 1:
            P_lm = -3.0 * cos_theta * sin_theta
            norm = jnp.sqrt(15.0 / (8.0 * PI))
        elif abs(m) == 2:
            P_lm = 3.0 * sin_theta**2
            norm = jnp.sqrt(15.0 / (32.0 * PI))
        else:
            P_lm = 0.0
            norm = 0.0
    elif l == 3:
        if m == 0:
            P_lm = 0.5 * (5.0 * cos_theta**3 - 3.0 * cos_theta)
            norm = jnp.sqrt(7.0 / (4.0 * PI))
        elif abs(m) == 1:
            P_lm = -1.5 * sin_theta * (5.0 * cos_theta**2 - 1.0)
            norm = jnp.sqrt(21.0 / (32.0 * PI))
        elif abs(m) == 2:
            P_lm = 15.0 * sin_theta**2 * cos_theta
            norm = jnp.sqrt(105.0 / (32.0 * PI))
        elif abs(m) == 3:
            P_lm = -15.0 * sin_theta**3
            norm = jnp.sqrt(35.0 / (64.0 * PI))
        else:
            P_lm = 0.0
            norm = 0.0
    elif l == 4:
        if m == 0:
            P_lm = (1.0/8.0) * (35.0 * cos_theta**4 - 30.0 * cos_theta**2 + 3.0)
            norm = 3.0 / (4.0 * jnp.sqrt(PI))
        elif abs(m) == 4:
            P_lm = 105.0 * sin_theta**4
            norm = 3.0 * jnp.sqrt(35.0 / (512.0 * PI))
        else:
            # Add other l=4 cases as needed
            P_lm = 0.0
            norm = 0.0
    else:
        P_lm = 0.0
        norm = 0.0
    
    # Apply sign convention for negative m
    if m < 0:
        P_lm *= (-1)**abs(m)
    
    # Combine with exponential
    result = norm * P_lm * jnp.exp(1j * m * phi)
    
    return result


def compute_chi_eff_chi_p(m1: float, m2: float, chi1: float, chi2: float, 
                          chi1_perp: float = 0.0, chi2_perp: float = 0.0) -> tuple:
    """
    Compute effective spin parameters.
    
    Args:
        m1, m2: Component masses
        chi1, chi2: Aligned spin components
        chi1_perp, chi2_perp: Perpendicular spin components
        
    Returns:
        chi_eff: Effective aligned spin
        chi_p: Effective precessing spin
    """
    q = m2 / m1  # Mass ratio (m2/m1 <= 1)
    
    # Effective aligned spin
    chi_eff = (chi1 + q * chi2) / (1.0 + q)
    
    # Effective precessing spin (approximate)
    chi_p = jnp.maximum(chi1_perp, (3.0 + 4.0*q)/(4.0 + 3.0*q) * q * chi2_perp)
    
    return chi_eff, chi_p


def compute_final_mass_spin(m1: float, m2: float, chi1: float, chi2: float) -> tuple:
    """
    Compute final black hole mass and spin using fitting formulas.
    
    Returns:
        M_final: Final mass
        a_final: Final dimensionless spin
    """
    eta = m1 * m2 / (m1 + m2)**2
    
    # Simplified final mass fit (should use more accurate formula)
    M_final = (m1 + m2) * (1.0 - 0.057 * eta)
    
    # Simplified final spin fit
    a_final = jnp.clip(chi1 + chi2, -0.998, 0.998)
    
    return M_final, a_final


# Cut-off frequency for IMRPhenomX modes
fM_CUT = 0.3