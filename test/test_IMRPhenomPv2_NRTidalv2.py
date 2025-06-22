"""
Test script for ripple IMRPhenomPv2_NRTidalv2 implementation.
"""

from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Import LALSuite
import lalsimulation as lalsim
import lal

# Import ripple components
import sys
import os
sys.path.append('/Users/Woute029/Documents/Code/jaxphm/ripple/src')

from ripplegw.waveforms.IMRPhenomPv2_NRTidalv2 import gen_IMRPhenomPv2_NRTidalv2_hphc
from ripplegw import Mc_eta_to_ms, get_match_arr, get_eff_pads


def test_IMRPhenomPv2_NRTidalv2():
    """
    Test ripple IMRPhenomPv2_NRTidalv2 implementation and compute waveform mismatch.
    """
    print("Testing IMRPhenomPv2_NRTidalv2 implementation...")
    
    # Test parameters for a typical BNS system
    # Masses in solar masses
    m1, m2 = 1.4, 1.3
    Mc = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    eta = m1 * m2 / (m1 + m2)**2
    
    # Spins (moderate precession)
    s1x, s1y, s1z = 0.00, 0.00, 0.00
    s2x, s2y, s2z = -0.00, 0.00, -0.00
    
    # Tidal parameters (typical for NS EOS)
    lambda1, lambda2 = 0.0, 0.0
    
    # Extrinsic parameters
    d_L = 100.0  # Mpc
    tc = 0.0     # s
    phi_ref = 0.0  # rad
    inclination = np.pi/4  # rad
    
    # Reference frequency
    f_ref = 50.0  # Hz
    
    # Frequency array
    f_min = 20.0
    f_max = 2048.0
    df = 1.0/128.0
    f = np.arange(f_min, f_max, df)
    
    # Parameters for ripple (using component masses directly, not lambda tildes)
    ripple_params = jnp.array([
        Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, 
        lambda1, lambda2, d_L, tc, phi_ref, inclination
    ])
    
    print(f"Test parameters:")
    print(f"  m1, m2 = {m1:.2f}, {m2:.2f} Msun")
    print(f"  Mc, eta = {Mc:.3f}, {eta:.4f}")
    print(f"  spins: s1=({s1x}, {s1y}, {s1z}), s2=({s2x}, {s2y}, {s2z})")
    print(f"  lambda1, lambda2 = {lambda1}, {lambda2}")
    print(f"  d_L = {d_L} Mpc")
    print(f"  inclination = {inclination:.2f} rad")
    print(f"  f_ref = {f_ref} Hz")
    print(f"  frequency range: {f_min} - {f_max} Hz")
    
    # Generate ripple waveform
    print("\nGenerating ripple waveform...")
    print(np.min(f), np.max(f))
    hp_ripple, hc_ripple = gen_IMRPhenomPv2_NRTidalv2_hphc(
        f, ripple_params, f_ref, use_lambda_tildes=False
    )
    
    # Generate LALSuite waveform for comparison
    print("\nGenerating LALSuite waveform for comparison...")
    
    # Convert parameters to LAL format
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    distance_m = d_L * 1e6 * lal.PC_SI
    
    # Generate waveform using LALSuite
    LALpars = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(LALpars, lambda1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(LALpars, lambda2)
    
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2_NRTidalv2")
    
    hp_lal, hc_lal = lalsim.SimInspiralChooseFDWaveform(
        m1_kg, m2_kg, 
        s1x, s1y, s1z, s2x, s2y, s2z,
        distance_m, inclination, phi_ref, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_ref,
        LALpars, 
        approximant
    )
    
    # Extract data and interpolate to ripple frequency grid
    hp_lal_data = hp_lal.data.data
    hc_lal_data = hc_lal.data.data
    f_lal = np.arange(len(hp_lal_data)) * df + f_min
    
    print(np.min(f_lal), np.max(f_lal))
    print(np.min(f), np.max(f))
    
    # Interpolate LAL data to ripple frequency grid
    hp_lal_interp = np.interp(f, f_lal, hp_lal_data, left=0, right=0)
    hc_lal_interp = np.interp(f, f_lal, hc_lal_data, left=0, right=0)
    
    # Calculate mismatch using ripple's get_match_arr function
    print("\nCalculating mismatch between Ripple and LALSuite...")
    # Create combined strain h = hp - i*hc
    h_ripple = hp_ripple - 1j * hc_ripple
    h_lal = hp_lal_interp - 1j * hc_lal_interp
    
    # Use a simple flat noise PSD for demonstration
    Sn = jnp.ones_like(f)  # Flat noise PSD
    
    # Get padding arrays for FFT
    pad_low, pad_high = get_eff_pads(f)
    
    # Calculate match
    match = get_match_arr(pad_low, pad_high, Sn, h_ripple, h_lal)
    mismatch = 1 - match
    
    print(f"  Match between Ripple and LALSuite: {match:.6f}")
    print(f"  Mismatch between Ripple and LALSuite: {mismatch:.6f}")
    
    # Create comparison plots
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('IMRPhenomPv2_NRTidalv2 Ripple vs LALSuite Comparison', fontsize=14)
    
    # Plot waveform amplitudes
    axes[0, 0].loglog(f, np.abs(hp_ripple), 'b-', label='hp (Ripple)', alpha=0.8)
    axes[0, 0].loglog(f, np.abs(hc_ripple), 'r-', label='hc (Ripple)', alpha=0.8)
    axes[0, 0].loglog(f, np.abs(hp_lal_interp), 'b--', label='hp (LALSuite)', alpha=0.8)
    axes[0, 0].loglog(f, np.abs(hc_lal_interp), 'r--', label='hc (LALSuite)', alpha=0.8)
    axes[0, 0].set_xlabel('Frequency [Hz]')
    axes[0, 0].set_ylabel('Strain amplitude')
    axes[0, 0].set_title('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot waveform phases
    axes[0, 1].semilogx(f, np.unwrap(np.angle(hp_ripple)), 'b-', label='hp (Ripple)', alpha=0.8)
    axes[0, 1].semilogx(f, np.unwrap(np.angle(hc_ripple)), 'r-', label='hc (Ripple)', alpha=0.8)
    axes[0, 1].semilogx(f, np.unwrap(np.angle(hp_lal_interp)), 'b--', label='hp (LALSuite)', alpha=0.8)
    axes[0, 1].semilogx(f, np.unwrap(np.angle(hc_lal_interp)), 'r--', label='hc (LALSuite)', alpha=0.8)
    axes[0, 1].set_xlabel('Frequency [Hz]')
    axes[0, 1].set_ylabel('Phase [rad]')
    axes[0, 1].set_title('Phase')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot amplitude difference
    hp_diff = np.abs(hp_ripple) - np.abs(hp_lal_interp)
    hc_diff = np.abs(hc_ripple) - np.abs(hc_lal_interp)
    axes[1, 0].semilogx(f, hp_diff / np.max(np.abs(hp_ripple)), 'b-', label='hp difference', alpha=0.8)
    axes[1, 0].semilogx(f, hc_diff / np.max(np.abs(hc_ripple)), 'r-', label='hc difference', alpha=0.8)
    axes[1, 0].set_xlabel('Frequency [Hz]')
    axes[1, 0].set_ylabel('Relative amplitude difference')
    axes[1, 0].set_title('Amplitude Difference (Ripple - LALSuite)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot phase difference
    phase_diff_hp = np.unwrap(np.angle(hp_ripple)) - np.unwrap(np.angle(hp_lal_interp))
    phase_diff_hc = np.unwrap(np.angle(hc_ripple)) - np.unwrap(np.angle(hc_lal_interp))
    axes[1, 1].semilogx(f, phase_diff_hp, 'b-', label='hp phase diff', alpha=0.8)
    axes[1, 1].semilogx(f, phase_diff_hc, 'r-', label='hc phase diff', alpha=0.8)
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('Phase difference [rad]')
    axes[1, 1].set_title('Phase Difference (Ripple - LALSuite)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('IMRPhenomPv2_NRTidalv2_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'IMRPhenomPv2_NRTidalv2_comparison.png'")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Ripple waveform: Successfully generated")
    print(f"  LALSuite waveform: Successfully generated")
    print(f"  Match between implementations: {match:.6f}")
    print(f"  Mismatch between implementations: {mismatch:.6f}")
    
    plt.show()


if __name__ == "__main__":
    test_IMRPhenomPv2_NRTidalv2()