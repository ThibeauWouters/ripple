#!/usr/bin/env python3
"""
Comparison script between JAX IMRPhenomXPHM and LALSuite IMRPhenomXPHM.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("🔬 IMRPhenomXPHM Comparison: JAX vs LALSuite")
    print("=" * 60)
    
    # Binary parameters
    f_min = 20.0
    f_max = 512.0
    delta_f = 0.5
    f = jnp.arange(f_min, f_max, delta_f)
    
    # Physical parameters
    m1 = 35.0      # Primary mass [solar masses]
    m2 = 25.0      # Secondary mass [solar masses] 
    chi1z = 0.5    # Primary aligned spin
    chi2z = -0.3   # Secondary aligned spin
    distance = 100.0  # Luminosity distance [Mpc]
    inclination = jnp.pi/3  # Inclination angle
    phi_ref = 0.0  # Reference phase
    f_ref = 20.0   # Reference frequency
    
    # Convert to chirp mass and eta for JAX implementation
    M_total = m1 + m2
    eta = (m1 * m2) / (M_total**2)
    Mc = M_total * (eta**(3.0/5.0))
    
    print(f"📊 Parameters:")
    print(f"   Masses: m1={m1} M☉, m2={m2} M☉")
    print(f"   Chirp mass: {Mc:.2f} M☉")
    print(f"   Mass ratio η: {eta:.3f}")
    print(f"   Spins: χ1z={chi1z}, χ2z={chi2z}")
    print(f"   Distance: {distance} Mpc")
    print(f"   Inclination: {float(inclination)*180/jnp.pi:.1f}°")
    print(f"   Frequency range: {f_min}-{f_max} Hz")
    
    # Generate JAX waveform
    print("\n🔄 Generating JAX IMRPhenomXPHM waveform...")
    try:
        from src.ripplegw.waveforms.IMRPhenomXPHM import gen_IMRPhenomXPHM
        
        params_jax = jnp.array([Mc, eta, chi1z, chi2z, distance, 0.0, phi_ref, inclination])
        hp_jax, hc_jax = gen_IMRPhenomXPHM(f, params_jax, f_ref)
        
        print(f"✅ JAX waveform generated")
        print(f"   hp amplitude: max = {jnp.max(jnp.abs(hp_jax)):.2e}")
        print(f"   hc amplitude: max = {jnp.max(jnp.abs(hc_jax)):.2e}")
        
    except Exception as e:
        print(f"❌ JAX waveform failed: {e}")
        hp_jax = jnp.zeros_like(f)
        hc_jax = jnp.zeros_like(f)
    
    # Generate LALSuite waveform 
    print("\n🔄 Generating LALSuite IMRPhenomXPHM waveform...")
    try:
        import lalsimulation as lalsim
        import lal
        
        # LALSuite parameters
        approximant = lalsim.IMRPhenomXPHM
        
        # Generate waveform
        hp_lal, hc_lal = lalsim.SimInspiralFD(
            m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
            0.0, 0.0, chi1z,  # S1x, S1y, S1z
            0.0, 0.0, chi2z,  # S2x, S2y, S2z
            distance * 1e6 * lal.PC_SI,  # distance in meters
            inclination, phi_ref,
            0.0, 0.0, 0.0,  # longAscNodes, eccentricity, meanPerAno
            delta_f, f_min, f_max, f_ref,
            None, approximant
        )
        
        # Extract arrays
        f_lal = np.array([hp_lal.f0 + i * hp_lal.deltaF for i in range(len(hp_lal.data.data))])
        hp_lal_data = np.array(hp_lal.data.data)
        hc_lal_data = np.array(hc_lal.data.data)
        
        # Interpolate to common frequency grid
        hp_lal_interp = np.interp(np.array(f), f_lal, hp_lal_data)
        hc_lal_interp = np.interp(np.array(f), f_lal, hc_lal_data)
        
        print(f"✅ LALSuite waveform generated")
        print(f"   hp amplitude: max = {np.max(np.abs(hp_lal_interp)):.2e}")
        print(f"   hc amplitude: max = {np.max(np.abs(hc_lal_interp)):.2e}")
        
        lal_available = True
        
    except ImportError:
        print("⚠️  LALSuite not available - generating reference waveform")
        # Create a mock reference waveform for demonstration
        v = (jnp.pi * f * M_total * 4.96e-6)**(1.0/3.0)  # Dimensionless velocity
        amp_22 = (Mc * 4.96e-6)**(5.0/6.0) / (jnp.pi**(2.0/3.0) * distance * 3.09e22) * (jnp.pi * f)**(-7.0/6.0)
        phase_22 = -jnp.cumsum(v**5) * 0.1
        
        hp_lal_interp = amp_22 * (1 + jnp.cos(inclination)**2) / 2 * jnp.cos(2 * phase_22)
        hc_lal_interp = amp_22 * jnp.cos(inclination) * jnp.sin(2 * phase_22)
        
        lal_available = False
        
    except Exception as e:
        print(f"❌ LALSuite waveform failed: {e}")
        hp_lal_interp = np.zeros_like(f)
        hc_lal_interp = np.zeros_like(f)
        lal_available = False
    
    # Create comparison plots
    print("\n📈 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('IMRPhenomXPHM Comparison: JAX vs LALSuite', fontsize=14, fontweight='bold')
    
    # Plot 1: Plus polarization comparison
    axes[0,0].loglog(f, jnp.abs(hp_jax), 'b-', label='JAX', linewidth=2, alpha=0.8)
    axes[0,0].loglog(f, np.abs(hp_lal_interp), 'r--', label='LALSuite' if lal_available else 'Reference', linewidth=2, alpha=0.8)
    axes[0,0].set_xlabel('Frequency [Hz]')
    axes[0,0].set_ylabel('|h₊|')
    axes[0,0].set_title('Plus Polarization')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Cross polarization comparison
    axes[0,1].loglog(f, jnp.abs(hc_jax), 'b-', label='JAX', linewidth=2, alpha=0.8)
    axes[0,1].loglog(f, np.abs(hc_lal_interp), 'r--', label='LALSuite' if lal_available else 'Reference', linewidth=2, alpha=0.8)
    axes[0,1].set_xlabel('Frequency [Hz]')
    axes[0,1].set_ylabel('|h₍|')
    axes[0,1].set_title('Cross Polarization')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Relative difference in plus polarization
    diff_hp = jnp.abs(hp_jax - hp_lal_interp) / (jnp.abs(hp_lal_interp) + 1e-30)
    axes[0,2].semilogx(f, diff_hp, 'g-', linewidth=2)
    axes[0,2].set_xlabel('Frequency [Hz]')
    axes[0,2].set_ylabel('Relative Difference')
    axes[0,2].set_title('|h₊| Relative Difference')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim([1e-3, 1e1])
    axes[0,2].set_yscale('log')
    
    # Plot 4: Phase comparison (unwrapped)
    phase_jax = jnp.unwrap(jnp.angle(hp_jax + 1j * hc_jax))
    phase_lal = np.unwrap(np.angle(hp_lal_interp + 1j * hc_lal_interp))
    
    axes[1,0].plot(f, phase_jax, 'b-', label='JAX', linewidth=2, alpha=0.8)
    axes[1,0].plot(f, phase_lal, 'r--', label='LALSuite' if lal_available else 'Reference', linewidth=2, alpha=0.8)
    axes[1,0].set_xlabel('Frequency [Hz]')
    axes[1,0].set_ylabel('Phase [rad]')
    axes[1,0].set_title('Phase Evolution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Overlap calculation
    # Compute frequency-domain overlap
    df = f[1] - f[0]
    inner_product = jnp.sum((jnp.conj(hp_jax + 1j * hc_jax) * (hp_lal_interp + 1j * hc_lal_interp)) * df)
    norm_jax = jnp.sqrt(jnp.sum(jnp.abs(hp_jax + 1j * hc_jax)**2 * df))
    norm_lal = jnp.sqrt(jnp.sum(jnp.abs(hp_lal_interp + 1j * hc_lal_interp)**2 * df))
    overlap = jnp.abs(inner_product) / (norm_jax * norm_lal)
    
    # Plot strain amplitude
    h_amp_jax = jnp.sqrt(hp_jax**2 + hc_jax**2)
    h_amp_lal = np.sqrt(hp_lal_interp**2 + hc_lal_interp**2)
    
    axes[1,1].loglog(f, h_amp_jax, 'b-', label='JAX', linewidth=2, alpha=0.8)
    axes[1,1].loglog(f, h_amp_lal, 'r--', label='LALSuite' if lal_available else 'Reference', linewidth=2, alpha=0.8)
    axes[1,1].set_xlabel('Frequency [Hz]')
    axes[1,1].set_ylabel('Total Strain |h|')
    axes[1,1].set_title(f'Total Strain (Overlap: {float(overlap):.3f})')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    axes[1,2].axis('off')
    summary_text = f"""
    Comparison Summary:
    
    JAX Implementation:
    • hp max: {jnp.max(jnp.abs(hp_jax)):.2e}
    • hc max: {jnp.max(jnp.abs(hc_jax)):.2e}
    
    {"LALSuite" if lal_available else "Reference"}:
    • hp max: {np.max(np.abs(hp_lal_interp)):.2e}
    • hc max: {np.max(np.abs(hc_lal_interp)):.2e}
    
    Overlap: {float(overlap):.3f}
    
    Status: {"✅ Good match" if overlap > 0.95 else "⚠️ Needs tuning"}
    """
    axes[1,2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', transform=axes[1,2].transAxes)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'IMRPhenomXPHM_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Comparison plot saved as: {output_file}")
    
    # Print summary
    print(f"\n📊 Comparison Results:")
    print(f"   Overlap: {float(overlap):.3f}")
    print(f"   Status: {'✅ Good agreement' if overlap > 0.95 else '⚠️ Implementation needs refinement'}")
    
    if overlap < 0.95:
        print(f"\n💡 Note: This is a simplified implementation.")
        print(f"   Full accuracy requires:")
        print(f"   • Complete mode set implementation")
        print(f"   • Accurate phenomenological coefficients")
        print(f"   • Proper NR calibration")
    
    plt.show()
    
    print("\n🎉 Comparison completed!")

if __name__ == "__main__":
    main()