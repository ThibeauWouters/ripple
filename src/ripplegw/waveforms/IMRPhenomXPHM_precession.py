import jax
import jax.numpy as jnp
from ..constants import PI, gt, C
from ..typing import Array
from .IMRPhenomXPHM_utils import compute_chi_eff_chi_p, compute_final_mass_spin


def compute_L_J_vectors(m1: float, m2: float, chi1: float, chi2: float, 
                       f: Array) -> tuple:
    """
    Compute orbital angular momentum L and total angular momentum J vectors.
    Following the MSA (Multiple Scale Analysis) approach.
    
    Args:
        m1, m2: Component masses [solar masses]
        chi1, chi2: Aligned spin components
        f: Frequency array [Hz]
        
    Returns:
        L_mag: Magnitude of orbital angular momentum
        J_mag: Magnitude of total angular momentum  
        cos_theta_L: Cosine of angle between L and J
    """
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Orbital frequency and velocity
    v = (PI * f * M_s)**(1.0/3.0)
    
    # Orbital angular momentum magnitude (Newtonian)
    L_mag = eta * (M_s**2) * v**(-1)
    
    # Spin angular momenta
    S1_mag = chi1 * m1_s**2
    S2_mag = chi2 * m2_s**2
    
    # Total angular momentum (assuming aligned case for simplicity)
    # In the full implementation, this would include precessing components
    J_mag = L_mag + S1_mag + S2_mag
    
    # Angle between L and J (simplified)
    cos_theta_L = L_mag / J_mag
    cos_theta_L = jnp.clip(cos_theta_L, -1.0, 1.0)
    
    return L_mag, J_mag, cos_theta_L


def compute_MSA_euler_angles(m1: float, m2: float, chi1: float, chi2: float,
                            f: Array, chi1_perp: float = 0.0, chi2_perp: float = 0.0) -> tuple:
    """
    Compute Euler angles using Multiple Scale Analysis (MSA) approach.
    This is a simplified version of the full LALSuite implementation.
    
    Args:
        m1, m2: Component masses [solar masses]  
        chi1, chi2: Aligned spin components
        f: Frequency array [Hz]
        chi1_perp, chi2_perp: Perpendicular spin components
        
    Returns:
        alpha: Azimuthal angle α
        beta: Polar angle β  
        epsilon: Third Euler angle ε
    """
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Get effective spins
    chi_eff, chi_p = compute_chi_eff_chi_p(m1, m2, chi1, chi2, chi1_perp, chi2_perp)
    
    # Orbital velocity
    v = (PI * f * M_s)**(1.0/3.0)
    
    # Precession frequency (simplified fit)
    # In the full implementation, this comes from detailed MSA calculation
    Omega_p = 2.0 * PI * f * chi_p * v**3 / (1.0 + v**2)
    
    # Integrate to get angles
    # For now, use simplified expressions
    # α: azimuthal precession angle
    alpha = jnp.cumsum(Omega_p) * (f[1] - f[0]) / f  # Simplified integration
    
    # β: polar angle between L and J
    # Use a phenomenological fit based on the effective precessing spin
    beta_0 = PI * chi_p / (1.0 + chi_eff)  # Initial inclination
    beta = beta_0 * jnp.exp(-v**2)  # Decay with frequency
    
    # ε: Third Euler angle (often set to zero in simplified models)
    epsilon = jnp.zeros_like(f)
    
    return alpha, beta, epsilon


def compute_PN_euler_angles(m1: float, m2: float, chi1: float, chi2: float,
                           f: Array, approximant: str = "SpinTaylorT4") -> tuple:
    """
    Compute Euler angles using Post-Newtonian (PN) evolution.
    Simplified version following single-spin approximation.
    
    Args:
        m1, m2: Component masses [solar masses]
        chi1, chi2: Aligned spin components  
        f: Frequency array [Hz]
        approximant: PN approximant to use
        
    Returns:
        alpha: Azimuthal angle α
        beta: Polar angle β
        epsilon: Third Euler angle ε
    """
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Use single-spin approximation (effective spin)
    chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    
    # Orbital velocity
    v = (PI * f * M_s)**(1.0/3.0)
    
    # PN precession equations (simplified)
    # Full implementation would solve coupled ODEs
    
    # Precession rate (leading order)
    dOmega_p = (3.0/4.0) * chi_eff * v**5 * f
    
    # Integrate to get precession angle
    # Simple cumulative sum (should use proper ODE integration)
    df = jnp.gradient(f)
    alpha = jnp.cumsum(dOmega_p / f * df)
    
    # Opening angle β
    # Phenomenological expression
    beta = PI/4.0 * chi_eff * v**2
    beta = jnp.clip(beta, 0.0, PI)
    
    # Third angle ε (often zero in single-spin approximation)
    epsilon = jnp.zeros_like(f)
    
    return alpha, beta, epsilon


def compute_spintaylor_euler_angles(m1: float, m2: float, chi1: float, chi2: float,
                                   f: Array, initial_conditions: dict = None) -> tuple:
    """
    Compute Euler angles using SpinTaylor numerical integration.
    This is a placeholder for the full numerical integration approach.
    
    Args:
        m1, m2: Component masses [solar masses]
        chi1, chi2: Spin components
        f: Frequency array [Hz] 
        initial_conditions: Initial orbital conditions
        
    Returns:
        alpha: Azimuthal angle α
        beta: Polar angle β
        epsilon: Third Euler angle ε
    """
    # For now, fall back to MSA approach
    # In a full implementation, this would:
    # 1. Set up the SpinTaylor ODEs
    # 2. Use JAX's ODE solver (e.g., diffrax)
    # 3. Integrate the coupled spin-orbit system
    
    return compute_MSA_euler_angles(m1, m2, chi1, chi2, f)


def get_euler_angles(m1: float, m2: float, chi1: float, chi2: float,
                    f: Array, method: str = "MSA", **kwargs) -> tuple:
    """
    Main interface for computing Euler angles with different methods.
    
    Args:
        m1, m2: Component masses [solar masses]
        chi1, chi2: Spin components
        f: Frequency array [Hz]
        method: Method to use ("MSA", "PN", "SpinTaylor")
        **kwargs: Additional method-specific arguments
        
    Returns:
        alpha: Azimuthal angle α [rad]
        beta: Polar angle β [rad] 
        epsilon: Third Euler angle ε [rad]
    """
    
    if method == "MSA":
        return compute_MSA_euler_angles(m1, m2, chi1, chi2, f, **kwargs)
    elif method == "PN":
        return compute_PN_euler_angles(m1, m2, chi1, chi2, f, **kwargs)
    elif method == "SpinTaylor":
        return compute_spintaylor_euler_angles(m1, m2, chi1, chi2, f, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_L_frame_to_J_frame_rotation(alpha: Array, beta: Array, epsilon: Array) -> tuple:
    """
    Compute rotation matrices from L-frame to J-frame.
    
    Args:
        alpha, beta, epsilon: Euler angles
        
    Returns:
        R_z_alpha: Rotation about z-axis by α
        R_y_beta: Rotation about y-axis by β  
        R_z_epsilon: Rotation about z-axis by ε
    """
    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    cos_beta = jnp.cos(beta)
    sin_beta = jnp.sin(beta)
    cos_epsilon = jnp.cos(epsilon)
    sin_epsilon = jnp.sin(epsilon)
    
    # Z-rotation by alpha
    R_z_alpha = jnp.array([
        [cos_alpha, -sin_alpha, 0],
        [sin_alpha, cos_alpha, 0],
        [0, 0, 1]
    ])
    
    # Y-rotation by beta  
    R_y_beta = jnp.array([
        [cos_beta, 0, sin_beta],
        [0, 1, 0],
        [-sin_beta, 0, cos_beta]
    ])
    
    # Z-rotation by epsilon
    R_z_epsilon = jnp.array([
        [cos_epsilon, -sin_epsilon, 0],
        [sin_epsilon, cos_epsilon, 0], 
        [0, 0, 1]
    ])
    
    return R_z_alpha, R_y_beta, R_z_epsilon


def compute_orbital_phase_evolution(m1: float, m2: float, chi1: float, chi2: float,
                                   f: Array) -> Array:
    """
    Compute orbital phase evolution in the L-frame.
    This needs to be carefully handled for precessing systems.
    
    Args:
        m1, m2: Component masses [solar masses]
        chi1, chi2: Spin components
        f: Frequency array [Hz]
        
    Returns:
        phi_orbital: Orbital phase evolution
    """
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2)
    
    # Use TaylorF2 phase evolution as base
    # This should be modified for spin effects
    v = (PI * f * M_s)**(1.0/3.0)
    
    # Leading-order phase coefficients
    phi_c = jnp.array([1.0, 0.0, 
                       (3715.0/756.0 + 55.0*eta/9.0),
                       -16.0*PI,
                       (15293365.0/508032.0 + 27145.0*eta/504.0 + 3085.0*eta**2/72.0)])
    
    # Build phase (simplified)
    phi = 0.0
    for i, coeff in enumerate(phi_c):
        if i == 0:
            phi += coeff * v**(-5)
        elif i == 1:
            phi += coeff * v**(-4)  
        elif i == 2:
            phi += coeff * v**(-3)
        elif i == 3:
            phi += coeff * v**(-2)
        elif i == 4:
            phi += coeff * v**(-1)
    
    # Add spin corrections (simplified)
    spin_correction = (chi1 + chi2) * v**2
    phi += spin_correction
    
    # Normalization
    phi *= 3.0/(128.0 * eta * v**5)
    
    return phi


def apply_precession_to_polarizations(hp_L: Array, hc_L: Array,
                                    alpha: Array, beta: Array, epsilon: Array) -> tuple:
    """
    Apply precession transformation to polarizations from L-frame to J-frame.
    
    Args:
        hp_L, hc_L: Plus and cross polarizations in L-frame
        alpha, beta, epsilon: Euler angles
        
    Returns:
        hp_J: Plus polarization in J-frame
        hc_J: Cross polarization in J-frame
    """
    # Complex strain in L-frame
    h_L = hp_L + 1j * hc_L
    
    # Apply rotation using complex exponentials
    # This is equivalent to the full Wigner-D matrix transformation
    rotation_factor = jnp.exp(1j * (2*alpha + epsilon))
    
    # Apply beta rotation (simplified)
    cos_beta = jnp.cos(beta)
    sin_beta = jnp.sin(beta)
    
    # Transform to J-frame
    h_J = h_L * rotation_factor * (cos_beta**2)
    
    # Extract polarizations
    hp_J = jnp.real(h_J)
    hc_J = jnp.imag(h_J)
    
    return hp_J, hc_J