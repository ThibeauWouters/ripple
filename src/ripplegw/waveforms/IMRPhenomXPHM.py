import jax
import jax.numpy as jnp
from ..constants import PI, gt, m_per_Mpc, C
from ..typing import Array
from .IMRPhenomXPHM_utils import wigner_d_matrix_element, spherical_harmonic_Ylm
from .IMRPhenomXPHM_precession import get_euler_angles
from .IMRPhenomXPHM_modes import get_nonprecessing_coprecessing_modes
from ripplegw import Mc_eta_to_ms


def twist_up_modes(
    modes_coprecessing: dict,
    alpha: Array,
    beta: Array,
    epsilon: Array,
    inclination: float,
    phi_ref: float = 0.0,
) -> tuple:
    """
    Core twist-up transformation from co-precessing L-frame to inertial J-frame.
    This is the heart of the IMRPhenomXPHM model.

    Args:
        modes_coprecessing: Dictionary of complex modes in L-frame
        alpha, beta, epsilon: Euler angles for precession
        inclination: Inclination angle of the binary [rad]
        phi_ref: Reference azimuthal angle [rad]

    Returns:
        hp: Plus polarization in J-frame
        hc: Cross polarization in J-frame
    """
    # Initialize output polarizations
    f_len = len(alpha)
    hp = jnp.zeros(f_len, dtype=complex)
    hc = jnp.zeros(f_len, dtype=complex)

    # Spherical harmonics for the detector frame
    # These depend on inclination and phi_ref
    Y_lm_detector = {}
    for l, m in modes_coprecessing.keys():
        Y_lm_detector[(l, m)] = spherical_harmonic_Ylm(l, m, inclination, phi_ref)
        # Also compute for negative m (symmetry relation)
        if m != 0:
            Y_lm_detector[(l, -m)] = (-1) ** m * jnp.conj(Y_lm_detector[(l, m)])

    # Twist-up transformation for each mode
    for l, m_prime in modes_coprecessing.keys():
        h_lm_coprecessing = modes_coprecessing[(l, m_prime)]

        # Sum over all m values for this l
        for m in range(-l, l + 1):
            if (l, m) not in Y_lm_detector:
                continue

            # Wigner d-matrix element
            d_lm_mp = wigner_d_matrix_element(l, m, m_prime, beta)

            # Phase factors from α and ε rotations
            phase_factor = jnp.exp(1j * (m * alpha + m_prime * epsilon))

            # Mode contribution in the J-frame
            h_lm_inertial = h_lm_coprecessing * d_lm_mp * phase_factor

            # Add to polarizations using spherical harmonics
            Y_lm = Y_lm_detector[(l, m)]

            # Plus and cross polarizations (Eqs. 3.5-3.7 in the reference)
            hp += h_lm_inertial * Y_lm
            hc += -1j * h_lm_inertial * Y_lm

            # Add negative m contribution (using mode symmetry)
            if m != 0:
                h_l_minus_m_inertial = (-1) ** l * jnp.conj(h_lm_inertial)
                Y_l_minus_m = Y_lm_detector[(l, -m)]

                hp += h_l_minus_m_inertial * Y_l_minus_m
                hc += -1j * h_l_minus_m_inertial * Y_l_minus_m

    return jnp.real(hp), jnp.real(hc)


def _gen_IMRPhenomXPHM(
    f: Array,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    f_ref: float,
    mode_list: list = None,
    precession_method: str = "MSA",
) -> tuple:
    """
    Internal function to generate IMRPhenomXPHM waveform.

    Args:
        f: Frequency array [Hz]
        theta_intrinsic: [m1, m2, chi1, chi2] in solar masses and dimensionless
        theta_extrinsic: [D, tc, phic, inclination]
        f_ref: Reference frequency [Hz]
        mode_list: List of (l,m) modes to include
        precession_method: Method for computing Euler angles

    Returns:
        hp: Plus polarization
        hc: Cross polarization
    """
    m1, m2, chi1, chi2 = theta_intrinsic
    D, tc, phic, inclination = theta_extrinsic

    # Default mode list for IMRPhenomXPHM
    if mode_list is None:
        mode_list = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]

    # Step 1: Generate non-precessing modes in co-precessing L-frame
    modes_coprecessing = get_nonprecessing_coprecessing_modes(
        f, m1, m2, chi1, chi2, D, mode_list
    )

    # Step 2: Compute Euler angles for precession
    alpha, beta, epsilon = get_euler_angles(
        m1, m2, chi1, chi2, f, method=precession_method
    )

    # Step 3: Apply twist-up transformation
    hp, hc = twist_up_modes(
        modes_coprecessing, alpha, beta, epsilon, inclination, phi_ref=0.0
    )

    # Step 4: Apply time and phase shifts
    # Time shift
    M_s = (m1 + m2) * gt
    time_shift = jnp.exp(2j * PI * f * tc)

    # Phase shift
    phase_shift = jnp.exp(1j * phic)

    # Reference phase adjustment
    # Use jnp.where to avoid boolean conversion issues in JIT
    idx_ref = jnp.argmin(jnp.abs(f - f_ref))
    ref_phase = jnp.angle(hp[idx_ref] + 1j * hc[idx_ref])
    phase_ref_correction = jnp.where(
        f_ref > 0, 
        jnp.exp(-1j * ref_phase),
        1.0 + 0j
    )

    # Apply all corrections
    correction = time_shift * phase_shift * phase_ref_correction
    hp_corrected = hp * jnp.real(correction) - hc * jnp.imag(correction)
    hc_corrected = hc * jnp.real(correction) + hp * jnp.imag(correction)

    return hp_corrected, hc_corrected


def gen_IMRPhenomXPHM(
    f: Array,
    params: Array,
    f_ref: float,
    mode_list: list = None,
    precession_method: str = "MSA",
) -> tuple:
    """
    Generate IMRPhenomXPHM frequency-domain waveform with precession and higher modes.

    This implements the precessing higher-mode extension of IMRPhenomX, following
    the methodology in arXiv:2004.06503 and the LALSuite implementation.

    Args:
        f: Frequency array [Hz]
        params: Parameter array [Mchirp, eta, chi1, chi2, D, tc, phic, inclination]
        f_ref: Reference frequency [Hz]
        mode_list: List of (l,m) tuples for modes to include. Default: [(2,2), (2,1), (3,3), (3,2), (4,4)]
        precession_method: Method for Euler angles ("MSA", "PN", "SpinTaylor")

    Parameter definitions:
        Mchirp: Chirp mass [solar masses]
        eta: Symmetric mass ratio [0 < eta <= 0.25]
        chi1: Dimensionless aligned spin of primary [-1 <= chi1 <= 1]
        chi2: Dimensionless aligned spin of secondary [-1 <= chi2 <= 1]
        D: Luminosity distance [Mpc]
        tc: Time of coalescence [s]
        phic: Phase of coalescence [rad]
        inclination: Inclination angle [0 <= iota <= π]

    Returns:
        hp: Plus polarization strain
        hc: Cross polarization strain

    Notes:
        - This is a simplified implementation focusing on the core algorithm
        - Full LALSuite version includes additional calibration and modes
        - Precession is handled via the "twist-up" procedure
        - Higher-order modes improve accuracy for asymmetric mass ratios
    """
    # Convert from Mc, eta to m1, m2
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))

    # Ensure m1 >= m2 convention
    m1, m2 = jnp.maximum(m1, m2), jnp.minimum(m1, m2)

    # Extract parameters
    theta_intrinsic = jnp.array([m1, m2, params[2], params[3]])
    theta_extrinsic = jnp.array([params[4], params[5], params[6], params[7]])

    # Generate the waveform
    hp, hc = _gen_IMRPhenomXPHM(
        f, theta_intrinsic, theta_extrinsic, f_ref, mode_list, precession_method
    )

    return hp, hc


def gen_IMRPhenomXPHM_one_mode(
    f: Array,
    params: Array,
    f_ref: float,
    l: int,
    m: int,
    precession_method: str = "MSA",
) -> Array:
    """
    Generate a single precessing mode for IMRPhenomXPHM.

    Args:
        f: Frequency array [Hz]
        params: Parameter array [Mchirp, eta, chi1, chi2, D, tc, phic, inclination]
        f_ref: Reference frequency [Hz]
        l, m: Mode indices
        precession_method: Method for Euler angles

    Returns:
        h_lm: Complex strain for the (l,m) mode
    """
    # Convert parameters
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    m1, m2 = jnp.maximum(m1, m2), jnp.minimum(m1, m2)
    chi1, chi2 = params[2], params[3]
    D, tc, phic, inclination = params[4], params[5], params[6], params[7]

    # Generate the single mode in co-precessing frame
    mode_list = [(l, m)]
    modes_coprecessing = get_nonprecessing_coprecessing_modes(
        f, m1, m2, chi1, chi2, D, mode_list
    )

    # Get Euler angles
    alpha, beta, epsilon = get_euler_angles(
        m1, m2, chi1, chi2, f, method=precession_method
    )

    # Apply twist-up for this specific mode
    h_lm_coprecessing = modes_coprecessing[(l, m)]

    # Transform to inertial frame
    d_lm = wigner_d_matrix_element(l, m, m, beta)
    phase_factor = jnp.exp(1j * (m * alpha + m * epsilon))
    h_lm_inertial = h_lm_coprecessing * d_lm * phase_factor

    # Apply time and phase shifts
    time_shift = jnp.exp(2j * PI * f * tc)
    phase_shift = jnp.exp(1j * phic)

    h_lm = h_lm_inertial * time_shift * phase_shift

    return h_lm


def gen_IMRPhenomXPHM_from_modes(
    f: Array, mode_dict: dict, inclination: float, phi_ref: float = 0.0
) -> tuple:
    """
    Generate polarizations from a pre-computed dictionary of modes.
    Useful for parameter estimation where modes can be cached.

    Args:
        f: Frequency array [Hz]
        mode_dict: Dictionary with (l,m) keys and complex mode values
        inclination: Inclination angle [rad]
        phi_ref: Reference azimuthal angle [rad]

    Returns:
        hp: Plus polarization
        hc: Cross polarization
    """
    # Initialize polarizations
    hp = jnp.zeros_like(f, dtype=float)
    hc = jnp.zeros_like(f, dtype=float)

    # Combine modes using spherical harmonics
    for (l, m), h_lm in mode_dict.items():
        Y_lm = spherical_harmonic_Ylm(l, m, inclination, phi_ref)

        # Add mode contribution
        hp += jnp.real(h_lm * Y_lm)
        hc += jnp.real(-1j * h_lm * Y_lm)

        # Add negative m contribution if m > 0
        if m > 0:
            Y_l_minus_m = spherical_harmonic_Ylm(l, -m, inclination, phi_ref)
            h_l_minus_m = (-1) ** l * jnp.conj(h_lm)

            hp += jnp.real(h_l_minus_m * Y_l_minus_m)
            hc += jnp.real(-1j * h_l_minus_m * Y_l_minus_m)

    return hp, hc


# JIT compile the main functions for performance
gen_IMRPhenomXPHM_jit = jax.jit(
    gen_IMRPhenomXPHM, static_argnames=["mode_list", "precession_method"]
)
gen_IMRPhenomXPHM_one_mode_jit = jax.jit(
    gen_IMRPhenomXPHM_one_mode, static_argnames=["l", "m", "precession_method"]
)


# Convenience function matching LALSuite interface
def XLALSimIMRPhenomXPHM(f: Array, params: Array, f_ref: float = 20.0) -> tuple:
    """
    LALSuite-style interface for IMRPhenomXPHM.

    Args:
        f: Frequency array [Hz]
        params: [Mchirp, eta, chi1, chi2, D, tc, phic, inclination]
        f_ref: Reference frequency [Hz]

    Returns:
        hp, hc: Plus and cross polarizations
    """
    return gen_IMRPhenomXPHM(f, params, f_ref)
