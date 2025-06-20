import numpy as np
import jax.numpy as jnp
from ripplegw.waveforms import gen_IMRPhenomXPHM, gen_IMRPhenomXPHM_jit
from ripplegw.waveforms.IMRPhenomXPHM_modes import get_nonprecessing_coprecessing_modes
from ripplegw.waveforms.IMRPhenomXPHM_precession import get_euler_angles
import pytest


def test_basic_IMRPhenomXPHM_generation():
    """Test basic functionality of IMRPhenomXPHM waveform generation."""
    
    # Basic test parameters
    f = jnp.linspace(20.0, 512.0, 1000)
    
    # Parameters: [Mchirp, eta, chi1, chi2, D, tc, phic, inclination]
    params = jnp.array([30.0, 0.24, 0.5, -0.3, 100.0, 0.0, 0.0, jnp.pi/3])
    f_ref = 20.0
    
    # Generate waveform
    hp, hc = gen_IMRPhenomXPHM(f, params, f_ref)
    
    # Basic checks
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert jnp.all(jnp.isfinite(hp))
    assert jnp.all(jnp.isfinite(hc))
    
    # Check that strain is not all zeros
    assert jnp.max(jnp.abs(hp)) > 1e-25
    assert jnp.max(jnp.abs(hc)) > 1e-25


def test_IMRPhenomXPHM_jit_compilation():
    """Test that JIT compilation works correctly."""
    
    f = jnp.linspace(20.0, 512.0, 100)
    params = jnp.array([20.0, 0.2, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    f_ref = 20.0
    
    # Test JIT compilation
    hp_jit, hc_jit = gen_IMRPhenomXPHM_jit(f, params, f_ref)
    hp_normal, hc_normal = gen_IMRPhenomXPHM(f, params, f_ref)
    
    # Results should be very close
    assert jnp.allclose(hp_jit, hp_normal, rtol=1e-12)
    assert jnp.allclose(hc_jit, hc_normal, rtol=1e-12)


def test_different_precession_methods():
    """Test different precession angle calculation methods."""
    
    f = jnp.linspace(20.0, 256.0, 200)
    params = jnp.array([25.0, 0.22, 0.3, 0.1, 100.0, 0.0, 0.0, jnp.pi/4])
    f_ref = 20.0
    
    # Test MSA method
    hp_msa, hc_msa = gen_IMRPhenomXPHM(f, params, f_ref, precession_method="MSA")
    
    # Test PN method  
    hp_pn, hc_pn = gen_IMRPhenomXPHM(f, params, f_ref, precession_method="PN")
    
    # Both should be finite and non-zero
    assert jnp.all(jnp.isfinite(hp_msa)) and jnp.all(jnp.isfinite(hc_msa))
    assert jnp.all(jnp.isfinite(hp_pn)) and jnp.all(jnp.isfinite(hc_pn))
    assert jnp.max(jnp.abs(hp_msa)) > 1e-25
    assert jnp.max(jnp.abs(hp_pn)) > 1e-25


def test_mode_selection():
    """Test selecting different sets of modes."""
    
    f = jnp.linspace(20.0, 256.0, 200)
    params = jnp.array([20.0, 0.25, 0.4, -0.2, 100.0, 0.0, 0.0, jnp.pi/6])
    f_ref = 20.0
    
    # Test with just (2,2) mode
    hp_22, hc_22 = gen_IMRPhenomXPHM(f, params, f_ref, mode_list=[(2,2)])
    
    # Test with multiple modes
    hp_multi, hc_multi = gen_IMRPhenomXPHM(f, params, f_ref, 
                                          mode_list=[(2,2), (2,1), (3,3)])
    
    # Multi-mode should generally have larger amplitude
    assert jnp.max(jnp.abs(hp_multi)) >= jnp.max(jnp.abs(hp_22))


def test_parameter_scaling():
    """Test that waveform scales correctly with distance."""
    
    f = jnp.linspace(20.0, 256.0, 100)
    f_ref = 20.0
    
    # Same system at different distances
    params_100Mpc = jnp.array([30.0, 0.24, 0.1, 0.1, 100.0, 0.0, 0.0, jnp.pi/4])
    params_200Mpc = jnp.array([30.0, 0.24, 0.1, 0.1, 200.0, 0.0, 0.0, jnp.pi/4])
    
    hp_100, hc_100 = gen_IMRPhenomXPHM(f, params_100Mpc, f_ref)
    hp_200, hc_200 = gen_IMRPhenomXPHM(f, params_200Mpc, f_ref)
    
    # Strain should scale inversely with distance
    ratio_hp = jnp.abs(hp_100) / jnp.abs(hp_200)
    ratio_hc = jnp.abs(hc_100) / jnp.abs(hc_200)
    
    # Check scaling (allowing for some numerical error)
    assert jnp.allclose(ratio_hp[jnp.abs(hp_100) > 1e-24], 2.0, rtol=0.1)
    assert jnp.allclose(ratio_hc[jnp.abs(hc_100) > 1e-24], 2.0, rtol=0.1)


def test_inclination_dependence():
    """Test that waveform changes correctly with inclination."""
    
    f = jnp.linspace(20.0, 256.0, 100)
    f_ref = 20.0
    
    # Face-on (iota = 0)
    params_faceon = jnp.array([25.0, 0.2, 0.2, 0.1, 100.0, 0.0, 0.0, 0.0])
    hp_faceon, hc_faceon = gen_IMRPhenomXPHM(f, params_faceon, f_ref)
    
    # Edge-on (iota = π/2)
    params_edgeon = jnp.array([25.0, 0.2, 0.2, 0.1, 100.0, 0.0, 0.0, jnp.pi/2])
    hp_edgeon, hc_edgeon = gen_IMRPhenomXPHM(f, params_edgeon, f_ref)
    
    # Face-on should have larger hp, smaller hc
    # Edge-on should have smaller hp, larger hc
    assert jnp.max(jnp.abs(hp_faceon)) > jnp.max(jnp.abs(hp_edgeon))
    assert jnp.max(jnp.abs(hc_edgeon)) > jnp.max(jnp.abs(hc_faceon))


def test_spin_effects():
    """Test that spin affects the waveform appropriately."""
    
    f = jnp.linspace(20.0, 256.0, 150)
    f_ref = 20.0
    
    # No spin
    params_nospin = jnp.array([30.0, 0.24, 0.0, 0.0, 100.0, 0.0, 0.0, jnp.pi/4])
    hp_nospin, hc_nospin = gen_IMRPhenomXPHM(f, params_nospin, f_ref)
    
    # With spin
    params_spin = jnp.array([30.0, 0.24, 0.6, -0.4, 100.0, 0.0, 0.0, jnp.pi/4])
    hp_spin, hc_spin = gen_IMRPhenomXPHM(f, params_spin, f_ref)
    
    # Waveforms should be different
    assert not jnp.allclose(hp_nospin, hp_spin, rtol=0.01)
    assert not jnp.allclose(hc_nospin, hc_spin, rtol=0.01)
    
    # Both should be finite
    assert jnp.all(jnp.isfinite(hp_spin)) and jnp.all(jnp.isfinite(hc_spin))


def test_euler_angles_calculation():
    """Test Euler angle calculation functions."""
    
    f = jnp.linspace(20.0, 256.0, 100)
    m1, m2 = 30.0, 20.0
    chi1, chi2 = 0.5, -0.3
    
    # Test MSA angles
    alpha_msa, beta_msa, epsilon_msa = get_euler_angles(m1, m2, chi1, chi2, f, "MSA")
    
    # Test PN angles
    alpha_pn, beta_pn, epsilon_pn = get_euler_angles(m1, m2, chi1, chi2, f, "PN")
    
    # All angles should be finite
    for angles in [(alpha_msa, beta_msa, epsilon_msa), (alpha_pn, beta_pn, epsilon_pn)]:
        for angle in angles:
            assert jnp.all(jnp.isfinite(angle))
            assert angle.shape == f.shape


def test_nonprecessing_modes():
    """Test generation of non-precessing modes."""
    
    f = jnp.linspace(20.0, 256.0, 100)
    m1, m2 = 25.0, 15.0
    chi1, chi2 = 0.3, 0.1
    D = 100.0
    
    # Generate modes
    modes = get_nonprecessing_coprecessing_modes(f, m1, m2, chi1, chi2, D)
    
    # Check that we get the expected modes
    expected_modes = [(2,2), (2,1), (3,3), (3,2), (4,4)]
    for mode in expected_modes:
        assert mode in modes
        assert modes[mode].shape == f.shape
        assert jnp.all(jnp.isfinite(modes[mode]))


def test_frequency_bounds():
    """Test behavior at frequency boundaries."""
    
    # Test very low frequencies
    f_low = jnp.linspace(5.0, 20.0, 50)
    params = jnp.array([40.0, 0.2, 0.1, 0.1, 100.0, 0.0, 0.0, jnp.pi/6])
    f_ref = 10.0
    
    hp_low, hc_low = gen_IMRPhenomXPHM(f_low, params, f_ref)
    assert jnp.all(jnp.isfinite(hp_low))
    assert jnp.all(jnp.isfinite(hc_low))
    
    # Test high frequencies
    f_high = jnp.linspace(500.0, 1000.0, 50)
    hp_high, hc_high = gen_IMRPhenomXPHM(f_high, params, f_ref)
    assert jnp.all(jnp.isfinite(hp_high))
    assert jnp.all(jnp.isfinite(hc_high))


def test_extreme_mass_ratios():
    """Test with extreme mass ratios."""
    
    f = jnp.linspace(20.0, 256.0, 100)
    f_ref = 20.0
    
    # Very unequal masses (q = 0.1, eta = 0.083)
    params_unequal = jnp.array([25.0, 0.083, 0.1, 0.1, 100.0, 0.0, 0.0, jnp.pi/4])
    hp_unequal, hc_unequal = gen_IMRPhenomXPHM(f, params_unequal, f_ref)
    
    # Equal masses (eta = 0.25)
    params_equal = jnp.array([25.0, 0.25, 0.1, 0.1, 100.0, 0.0, 0.0, jnp.pi/4])
    hp_equal, hc_equal = gen_IMRPhenomXPHM(f, params_equal, f_ref)
    
    # Both should be finite
    assert jnp.all(jnp.isfinite(hp_unequal)) and jnp.all(jnp.isfinite(hc_unequal))
    assert jnp.all(jnp.isfinite(hp_equal)) and jnp.all(jnp.isfinite(hc_equal))
    
    # Unequal mass system should show different behavior (higher modes more important)
    assert not jnp.allclose(hp_unequal, hp_equal, rtol=0.1)


if __name__ == "__main__":
    # Run some basic tests if executed directly
    test_basic_IMRPhenomXPHM_generation()
    test_IMRPhenomXPHM_jit_compilation()
    test_different_precession_methods()
    print("All basic tests passed!")