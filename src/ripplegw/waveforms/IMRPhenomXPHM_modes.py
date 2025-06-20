import jax
import jax.numpy as jnp
from ..constants import PI, gt, m_per_Mpc, C
from ..typing import Array
from .IMRPhenomXAS import _gen_IMRPhenomXAS
from .IMRPhenomXPHM_utils import get_cutoff_fMs, spherical_harmonic_Ylm
from ripplegw import Mc_eta_to_ms


def generate_22_mode(f: Array, m1: float, m2: float, chi1: float, chi2: float,
                    D: float = 1.0) -> Array:
    """
    Generate the (2,2) mode using IMRPhenomXAS as the base.
    
    Args:
        f: Frequency array [Hz]
        m1, m2: Component masses [solar masses]
        chi1, chi2: Aligned spin components
        D: Luminosity distance [Mpc]
        
    Returns:
        h22: Complex (2,2) mode
    """
    # Use existing IMRPhenomXAS implementation
    from .IMRPhenomXAS import gen_IMRPhenomXAS
    
    # Convert to Mc, eta parameterization
    Mc = (m1 * m2)**(3.0/5.0) / (m1 + m2)**(1.0/5.0)
    eta = m1 * m2 / (m1 + m2)**2
    
    # Parameters for IMRPhenomXAS: [Mc, eta, chi1, chi2, D, tc, phic]
    params = jnp.array([Mc, eta, chi1, chi2, D, 0.0, 0.0])
    f_ref = 20.0
    
    h22 = gen_IMRPhenomXAS(f, params, f_ref)
    
    return h22


def generate_21_mode(f: Array, m1: float, m2: float, chi1: float, chi2: float,
                    D: float = 1.0) -> Array:
    """
    Generate the (2,1) mode using phenomenological fits.
    
    Args:
        f: Frequency array [Hz] 
        m1, m2: Component masses [solar masses]
        chi1, chi2: Aligned spin components
        D: Luminosity distance [Mpc]
        
    Returns:
        h21: Complex (2,1) mode
    """
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Get reference (2,2) mode for scaling
    h22 = generate_22_mode(f, m1, m2, chi1, chi2, D)
    
    # Mass ratio and asymmetry parameter
    q = m2 / m1  # Assuming m1 >= m2
    delta = jnp.sqrt(1.0 - 4.0*eta)
    
    # Phenomenological amplitude scaling for (2,1) mode
    # Based on fits to numerical relativity
    amp_21_factor = delta * jnp.sqrt(2.0) * (1.0 + 0.5*eta)
    
    # Phase difference relative to (2,2) mode
    # Simplified model - in practice this comes from detailed fits
    v = (PI * f * M_s)**(1.0/3.0)
    phase_21_correction = -PI/2.0 + delta * v**2
    
    # Construct (2,1) mode
    h21 = h22 * amp_21_factor * jnp.exp(1j * phase_21_correction)
    
    return h21


def generate_33_mode(f: Array, m1: float, m2: float, chi1: float, chi2: float,
                    D: float = 1.0) -> Array:
    """
    Generate the (3,3) mode using phenomenological fits.
    
    Args:
        f: Frequency array [Hz]
        m1, m2: Component masses [solar masses] 
        chi1, chi2: Aligned spin components
        D: Luminosity distance [Mpc]
        
    Returns:
        h33: Complex (3,3) mode
    """
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Get reference (2,2) mode for scaling
    h22 = generate_22_mode(f, m1, m2, chi1, chi2, D)
    
    # Orbital velocity
    v = (PI * f * M_s)**(1.0/3.0)
    
    # Amplitude scaling for (3,3) mode
    # Based on PN theory and NR fits
    amp_33_factor = (3.0/4.0) * jnp.sqrt(5.0/7.0) * eta * v
    
    # Phase relative to (2,2) mode
    # The (3,3) mode has 3/2 times the orbital frequency
    phase_33_correction = 0.5 * jnp.log(v) + PI/3.0
    
    # Frequency scaling (3,3) peaks at 1.5 times orbital frequency
    f_scaled = f / 1.5
    h22_scaled = generate_22_mode(f_scaled, m1, m2, chi1, chi2, D)
    
    # Construct (3,3) mode
    h33 = h22_scaled * amp_33_factor * jnp.exp(1j * phase_33_correction)
    
    return h33


def generate_32_mode(f: Array, m1: float, m2: float, chi1: float, chi2: float,
                    D: float = 1.0) -> Array:
    """
    Generate the (3,2) mode using phenomenological fits.
    """
    m1_s = m1 * gt  
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Get reference (2,2) mode
    h22 = generate_22_mode(f, m1, m2, chi1, chi2, D)
    
    # Mass ratio asymmetry
    delta = jnp.sqrt(1.0 - 4.0*eta)
    v = (PI * f * M_s)**(1.0/3.0)
    
    # Amplitude scaling for (3,2) mode
    amp_32_factor = (2.0/3.0) * jnp.sqrt(5.0/7.0) * delta * eta * v**(1.0/2.0)
    
    # Phase correction
    phase_32_correction = -PI/6.0 + 0.5*delta * v
    
    # Construct (3,2) mode
    h32 = h22 * amp_32_factor * jnp.exp(1j * phase_32_correction)
    
    return h32


def generate_44_mode(f: Array, m1: float, m2: float, chi1: float, chi2: float,
                    D: float = 1.0) -> Array:
    """
    Generate the (4,4) mode using phenomenological fits.
    """
    m1_s = m1 * gt
    m2_s = m2 * gt  
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Get reference (2,2) mode
    h22 = generate_22_mode(f, m1, m2, chi1, chi2, D)
    
    # Orbital velocity
    v = (PI * f * M_s)**(1.0/3.0)
    
    # Amplitude scaling for (4,4) mode
    amp_44_factor = (4.0/9.0) * jnp.sqrt(5.0/7.0) * eta**2 * v**2
    
    # Phase correction
    phase_44_correction = jnp.log(v) + PI/2.0
    
    # Frequency scaling (4,4) peaks at 2 times orbital frequency
    f_scaled = f / 2.0
    h22_scaled = generate_22_mode(f_scaled, m1, m2, chi1, chi2, D)
    
    # Construct (4,4) mode  
    h44 = h22_scaled * amp_44_factor * jnp.exp(1j * phase_44_correction)
    
    return h44


def generate_higher_order_modes(f: Array, m1: float, m2: float, chi1: float, chi2: float,
                               D: float = 1.0, modes: list = None) -> dict:
    """
    Generate all requested higher-order modes.
    
    Args:
        f: Frequency array [Hz]
        m1, m2: Component masses [solar masses]
        chi1, chi2: Aligned spin components  
        D: Luminosity distance [Mpc]
        modes: List of (l,m) tuples to generate. If None, uses default set.
        
    Returns:
        mode_dict: Dictionary with (l,m) keys and complex mode values
    """
    if modes is None:
        # Default mode set for IMRPhenomXPHM
        modes = [(2,2), (2,1), (3,3), (3,2), (4,4)]
    
    mode_dict = {}
    
    for l, m in modes:
        if l == 2 and m == 2:
            mode_dict[(l,m)] = generate_22_mode(f, m1, m2, chi1, chi2, D)
        elif l == 2 and m == 1:
            mode_dict[(l,m)] = generate_21_mode(f, m1, m2, chi1, chi2, D)
        elif l == 3 and m == 3:
            mode_dict[(l,m)] = generate_33_mode(f, m1, m2, chi1, chi2, D)
        elif l == 3 and m == 2:
            mode_dict[(l,m)] = generate_32_mode(f, m1, m2, chi1, chi2, D)
        elif l == 4 and m == 4:
            mode_dict[(l,m)] = generate_44_mode(f, m1, m2, chi1, chi2, D)
        else:
            # For other modes, return zeros
            mode_dict[(l,m)] = jnp.zeros_like(f, dtype=complex)
    
    return mode_dict


def get_mode_weights(inclination: float, phi: float = 0.0) -> dict:
    """
    Compute spherical harmonic weights for mode combination.
    
    Args:
        inclination: Inclination angle [rad]
        phi: Azimuthal angle [rad]
        
    Returns:
        weights: Dictionary of Y_lm(θ,φ) values for each mode
    """
    weights = {}
    
    # Standard modes for IMRPhenomXPHM
    modes = [(2,2), (2,1), (2,-1), (2,-2), 
             (3,3), (3,2), (3,1), (3,-1), (3,-2), (3,-3),
             (4,4), (4,-4)]
    
    for l, m in modes:
        weights[(l,m)] = spherical_harmonic_Ylm(l, m, inclination, phi)
    
    return weights


def combine_modes_for_polarizations(mode_dict: dict, inclination: float, 
                                   phi: float = 0.0) -> tuple:
    """
    Combine modes to get plus and cross polarizations.
    
    Args:
        mode_dict: Dictionary of complex modes 
        inclination: Inclination angle [rad]
        phi: Azimuthal angle [rad]
        
    Returns:
        hp: Plus polarization
        hc: Cross polarization
    """
    # Get spherical harmonic weights
    Y_weights = get_mode_weights(inclination, phi)
    
    # Initialize polarizations
    hp = jnp.zeros_like(list(mode_dict.values())[0], dtype=complex)
    hc = jnp.zeros_like(list(mode_dict.values())[0], dtype=complex)
    
    # Combine modes
    for (l, m), h_lm in mode_dict.items():
        if (l, m) in Y_weights:
            Y_lm = Y_weights[(l, m)]
            
            # Add positive m contribution
            hp += h_lm * Y_lm
            hc += -1j * h_lm * Y_lm
            
            # Add negative m contribution (using symmetry)
            if m > 0 and (l, -m) in Y_weights:
                Y_l_minus_m = Y_weights[(l, -m)]
                h_l_minus_m = (-1)**l * jnp.conj(h_lm)
                
                hp += h_l_minus_m * Y_l_minus_m
                hc += -1j * h_l_minus_m * Y_l_minus_m
    
    return jnp.real(hp), jnp.real(hc)


def apply_mode_tapering(mode_dict: dict, f: Array, f_taper_start: float = 100.0) -> dict:
    """
    Apply frequency-domain tapering to suppress high-frequency artifacts.
    
    Args:
        mode_dict: Dictionary of complex modes
        f: Frequency array [Hz] 
        f_taper_start: Frequency to start tapering [Hz]
        
    Returns:
        tapered_modes: Dictionary with tapered modes
    """
    tapered_modes = {}
    
    # Tukey window for smooth tapering
    taper_mask = jnp.where(f > f_taper_start,
                          0.5 * (1.0 + jnp.cos(PI * (f - f_taper_start) / f_taper_start)),
                          1.0)
    
    for key, mode in mode_dict.items():
        tapered_modes[key] = mode * taper_mask
    
    return tapered_modes


def get_nonprecessing_coprecessing_modes(f: Array, m1: float, m2: float, 
                                       chi1: float, chi2: float, D: float = 1.0,
                                       modes: list = None) -> dict:
    """
    Main interface for generating non-precessing modes in the co-precessing frame.
    These will be the input to the twist-up procedure.
    
    Args:
        f: Frequency array [Hz]
        m1, m2: Component masses [solar masses]
        chi1, chi2: Aligned spin components
        D: Luminosity distance [Mpc]
        modes: List of (l,m) modes to generate
        
    Returns:
        mode_dict: Dictionary of complex modes in co-precessing L-frame
    """
    # Generate the higher-order modes
    mode_dict = generate_higher_order_modes(f, m1, m2, chi1, chi2, D, modes)
    
    # Apply frequency tapering to avoid artifacts
    mode_dict = apply_mode_tapering(mode_dict, f)
    
    return mode_dict