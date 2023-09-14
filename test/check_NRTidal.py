import jax.numpy as jnp
import jax
import pandas as pd

from ripple import get_eff_pads, get_match_arr
from tqdm import tqdm

import numpy as np
from ripple import ms_to_Mc_eta, Mc_eta_to_ms
import lalsimulation as lalsim
import lal

import matplotlib.pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)


def random_match_NRTidal(n, IMRphenom = "IMRPhenomD"):
    """
    Generates random wavefporm match scores between LAL and ripple.
    Args:
        n: number of matches to be made
        IMRphenom: str
            string indicating which waveform for the underlying BBH strain we wish to use

    Returns:
        TODO
    """

    from ripple.waveforms.X_NRTidalv2 import gen_NRTidalv2_hphc as waveform_generator

    # Specify frequency range
    f_l = 16
    f_sampling = 2 * 4096

    # TODO - check at higher frequency
    f_u = f_sampling // 2
    T = 16

    # Build the frequency grid
    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = np.fft.rfftfreq(tlen, delta_t)
    df = freqs[1] - freqs[0]
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    f_ref = f_l

    @jax.jit
    def waveform(theta):
        hp, _ = waveform_generator(fs, theta, f_ref, IMRphenom=IMRphenom)
        return hp

    # Get a frequency domain waveform
    thetas = []
    matches = []
    f_ASD, ASD = np.loadtxt("O3Livingston.txt", unpack=True)

    # TODO - check NRTidal with precession
    if "PhenomP" in IMRphenom:
        for i in tqdm(range(n)):
            precessing_matchmaking(
                IMRphenom, f_l, f_u, df, fs, waveform, f_ASD, ASD, thetas, matches
            )
    else:
        for i in tqdm(range(n)):
            non_precessing_matchmaking(
                IMRphenom, f_l, f_u, df, fs, waveform, f_ASD, ASD, thetas, matches
            )

    thetas = np.array(thetas)
    matches = np.array(matches)

    df = save_matches("NRTidal_matches.csv", thetas, matches)

    mismatches = np.log10(1 - matches)
    print("Mean match:", np.mean(matches))
    print("Median match:", np.median(matches))
    print("Minimum match:", np.min(matches))

    print("------------------------")

    print("Mean mismatch:", np.mean(mismatches))
    print("Median mismatch:", np.median(mismatches))
    print("Minimum mismatch:", np.min(mismatches))

    return df


def non_precessing_matchmaking(
    IMRphenom, f_l, f_u, df, fs, waveform, f_ASD, ASD, thetas, matches
):

    # These ranges are taken from: https://wiki.ligo.org/CBC/Waveforms/WaveformTable
    m_l, m_u = 1.0, 3.0
    chi_l, chi_u = -0.7, 0.7
    lambda_u = 5000

    m1 = np.random.uniform(m_l, m_u)
    m2 = np.random.uniform(m_l, m_u)
    s1 = np.random.uniform(chi_l, chi_u)
    s2 = np.random.uniform(chi_l, chi_u)
    l1 = np.random.uniform(0, lambda_u)
    l2 = np.random.uniform(0, lambda_u)

    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = 0 # np.pi / 2.0con
    phi_ref = 0.0

    if m1 < m2:
        theta = np.array([m2, m1, s2, s1, l2, l1, dist_mpc, tc, phic, inclination])
    elif m1 > m2:
        theta = np.array([m1, m2, s1, s2, l1, l2, dist_mpc, tc, phic, inclination])
    else:
        raise ValueError("Something went wrong with the parameters")
    approximant = lalsim.SimInspiralGetApproximantFromString(IMRphenom)

    f_ref = f_l
    m1_kg = theta[0] * lal.MSUN_SI
    m2_kg = theta[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI

    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        theta[2],
        0.0,
        0.0,
        theta[3],
        distance,
        inclination,
        phi_ref,
        0,
        l1,
        l2,
        df,
        f_l,
        f_u,
        f_ref,
        None,
        approximant,
    )

    freqs_lal = np.arange(len(hp.data.data)) * df
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

    theta_ripple = jnp.array(
        [Mc, eta, theta[2], theta[3], l1, l2, dist_mpc, tc, phic, inclination]
    )
    hp_ripple = waveform(theta_ripple)
    pad_low, pad_high = get_eff_pads(fs)
    PSD_vals = np.interp(fs, f_ASD, ASD) ** 2

    mask_lal = (freqs_lal > f_l) & (freqs_lal < f_u)
    hp_lalsuite = hp.data.data[mask_lal]
    matches.append(
        get_match_arr(
            pad_low,
            pad_high,
            # np.ones_like(fs) * 1.0e-42,
            PSD_vals,
            hp_ripple,
            hp_lalsuite,
        )
    )
    thetas.append(theta)


def precessing_matchmaking(
    IMRphenom, f_l, f_u, df, fs, waveform, f_ASD, ASD, thetas, matches
):
    m1 = np.random.uniform(1.0, 100.0)
    m2 = np.random.uniform(1.0, 100.0)
    s1_amp = np.random.uniform(0.0, 1.0)
    s2_amp = np.random.uniform(0.0, 1.0)
    s1_phi = np.random.uniform(0, 2 * np.pi)
    s2_phi = np.random.uniform(0, 2 * np.pi)
    s1_thetahelper = np.random.uniform(0, 1)
    s2_thetahelper = np.random.uniform(0, 1)
    s1_theta = np.arccos(1 - 2 * s1_thetahelper)
    s2_theta = np.arccos(1 - 2 * s2_thetahelper)
    # translate that into cartesian
    s1x = s1_amp * np.sin(s1_theta) * np.cos(s1_phi)
    s1y = s1_amp * np.sin(s1_theta) * np.sin(s1_phi)
    s1z = s1_amp * np.cos(s1_theta)

    s2x = s2_amp * np.sin(s2_theta) * np.cos(s2_phi)
    s2y = s2_amp * np.sin(s2_theta) * np.sin(s2_phi)
    s2z = s2_amp * np.cos(s2_theta)

    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = np.pi / 2.0
    phi_ref = 0

    if m1 < m2:
        theta = np.array(
            [m2, m1, s2x, s2y, s2z, s1x, s1y, s1z, dist_mpc, tc, phi_ref, inclination]
        )
    elif m1 > m2:
        theta = np.array(
            [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phi_ref, inclination]
        )
    else:
        raise ValueError("Something went wrong with the parameters")
    approximant = lalsim.SimInspiralGetApproximantFromString(IMRphenom)

    f_ref = f_l
    m1_kg = theta[0] * lal.MSUN_SI
    m2_kg = theta[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI

    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        theta[2],
        theta[3],
        theta[4],
        theta[5],
        theta[6],
        theta[7],
        distance,
        inclination,
        phi_ref,
        0,
        0.0,
        0.0,
        df,
        f_l,
        f_u,
        f_ref,
        None,
        approximant,
    )
    freqs_lal = np.arange(len(hp.data.data)) * df

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    theta_ripple = np.array(
        [
            Mc,
            eta,
            theta[2],
            theta[3],
            theta[4],
            theta[5],
            theta[6],
            theta[7],
            dist_mpc,
            tc,
            phic,
            inclination,
        ]
    )
    hp_ripple = waveform(theta_ripple)
    pad_low, pad_high = get_eff_pads(fs)
    PSD_vals = np.interp(fs, f_ASD, ASD) ** 2

    mask_lal = (freqs_lal > f_l) & (freqs_lal < f_u)
    hp_lalsuite = hp.data.data[mask_lal]
    matches.append(
        get_match_arr(
            pad_low,
            pad_high,
            # np.ones_like(fs) * 1.0e-42,
            PSD_vals,
            hp_ripple,
            hp_lalsuite,
        )
    )
    thetas.append(theta)


def save_matches(filename, thetas, matches):
    # header = ["m1", "m2", "chi1", "chi2", "lambda1", "lambda2", "match"]

    m1      = thetas[:, 0]
    m2      = thetas[:, 1]
    chi1    = thetas[:, 2]
    chi2    = thetas[:, 3]
    lambda1 = thetas[:, 4]
    lambda2 = thetas[:, 5]

    my_dict = {'m1': m1, 'm2': m2, 'chi1': chi1, 'chi2': chi2, 'lambda1': lambda1, 'lambda2': lambda2, 'match': matches}

    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv(filename)

    return df




if __name__ == "__main__":
    # Choose from "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomPv2"
    df = random_match_NRTidal(1000)

    print(df)
