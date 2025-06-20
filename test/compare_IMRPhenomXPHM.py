"""
Comparison script between JAX IMRPhenomXPHM and LALSuite IMRPhenomXPHM.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from ripplegw.waveforms.IMRPhenomXPHM import gen_IMRPhenomXPHM
import lalsimulation as lalsim
import lal

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision for accuracy

def main():
    print("🔬 IMRPhenomXPHM Comparison: JAX vs LALSuite")
    print("=" * 60)
    
    # Binary parameters
    f_min = 20.0
    f_max = 1024.0
    T = 4
    # delta_f = 1/T
    delta_f = 1/16
    f = jnp.arange(f_min, f_max, delta_f)
    
    # Physical parameters
    m1 = 35.0      # Primary mass [solar masses]
    m2 = 25.0      # Secondary mass [solar masses] 
    
    chi1x = 0.0    # Primary aligned spin x # FIXME: fix this
    chi1y = 0.0    # Primary aligned spin y # FIXME: fix this
    chi1z = 0.0    # Primary aligned spin z
    
    chi2x = -0.0   # Secondary aligned spin x # FIXME: fix this
    chi2y = -0.0   # Secondary aligned spin y # FIXME: fix this
    chi2z = -0.0   # Secondary aligned spin z
    
    distance = 440.0  # Luminosity distance [Mpc]
    inclination = jnp.pi/3  # Inclination angle
    phi_ref = 0.0  # Reference phase
    f_ref = 20.0   # Reference frequency
    
    # TODO: 
    # hp, _ = lalsim.SimInspiralChooseFDWaveform(
    #     m1_kg,
    #     m2_kg,
    #     chi1x,
    #     chi1y,
    #     chi1z,
    #     chi2x,
    #     chi2y,
    #     chi2z,
    #     distance,
    #     inclination,
    #     phic,
    #     0,
    #     0,
    #     0,
    #     df,
    #     f_l,
    #     f_u,
    #     f_ref,
    #     laldict,
    #     approximant,
    # )
    
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
    params_jax = jnp.array([Mc, eta, chi1z, chi2z, distance, 0.0, phi_ref, inclination])
    hp_jax, hc_jax = gen_IMRPhenomXPHM(f,
                                       params_jax,
                                       f_ref,
                                       mode_list=[(2, 2)],
                                       precession_method="SpinTaylor")
    
    print(f"✅ JAX waveform generated")
    print(f"   hp amplitude: max = {jnp.max(jnp.abs(hp_jax)):.2e}")
    print(f"   hc amplitude: max = {jnp.max(jnp.abs(hc_jax)):.2e}")
        
    # Generate LALSuite waveform 
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXPHM")
    
    # Generate waveform
    hp_lal, hc_lal = lalsim.SimInspiralFD(m1 * lal.MSUN_SI,
                                          m2 * lal.MSUN_SI,
                                          chi1x,
                                          chi1y,
                                          chi1z,
                                          chi2x,
                                          chi2y,
                                          chi2z,
                                          distance * 1e6 * lal.PC_SI,  # distance in meters
                                          inclination,
                                          phi_ref,
                                          0.0, 0.0, 0.0,  # longAscNodes, eccentricity, meanPerAno
                                          delta_f,
                                          f_min,
                                          f_max,
                                          f_ref,
                                          None,
                                          approximant
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
    
    # Create comparison plots
    print("\n📈 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('IMRPhenomXPHM Comparison: JAX vs LALSuite', fontsize=14, fontweight='bold')
    
    # Plot 1: Plus polarization comparison
    axes[0,0].loglog(f, jnp.abs(hp_jax), 'b-', label='JAX', linewidth=2, alpha=0.8)
    axes[0,0].loglog(f, np.abs(hp_lal_interp), 'r--', label='LALSuite', linewidth=2, alpha=0.8)
    axes[0,0].set_xlabel('Frequency [Hz]')
    axes[0,0].set_ylabel('|h₊|')
    axes[0,0].set_title('Plus Polarization')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Cross polarization comparison
    axes[0,1].loglog(f, jnp.abs(hc_jax), 'b-', label='JAX', linewidth=2, alpha=0.8)
    axes[0,1].loglog(f, np.abs(hc_lal_interp), 'r--', label='LALSuite', linewidth=2, alpha=0.8)
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
    axes[1,0].plot(f, phase_lal, 'r--', label='LALSuite', linewidth=2, alpha=0.8)
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
    axes[1,1].loglog(f, h_amp_lal, 'r--', label='LALSuite', linewidth=2, alpha=0.8)
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
    
    "LALSuite":
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
    
    print("\n🎉 Comparison completed!")
    
    ### Thibeau
    a = 0.75
    plt.subplots(1, 1, figsize=(17, 7))
    # plt.subplot(1, 2, 1)

    print(type(hp_lal))
    # print(np.shape(hp_lal))

    plt.plot(f, hp_jax, "-", label = "ripple", alpha = a)
    plt.plot(f, hp_lal_interp, "-", label = "LAL", alpha = a)
    plt.xlabel("Frequency")
    plt.ylabel("Strain")
    plt.xscale('log')
    plt.legend()
    plt.savefig("IMRPhenomXPHM_comparison_TW.png", dpi=300)

if __name__ == "__main__":
    main()