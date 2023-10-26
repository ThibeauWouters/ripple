import jax
import jax.numpy as jnp

from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms


# Phase coefficients

def get_qm_def(lambda_):
    # TODO have to implement this!  
    return 1



def get_3PNSOCoeff(mByM):
    return  mByM * (25. + 38./3. * mByM)

def get_5PNSOCoeff(mByM):
    return -mByM*(1391.5/8.4 - mByM * (1. - mByM) * 10./3. + mByM * (1276./8.1 + mByM * (1. - mByM) * 170./9.))

def get_7PNSOCoeff(mByM):
    
    eta = mByM*(1.-mByM)
    return mByM * (-17097.8035/4.8384 + eta * 28764.25/6.72 + eta * eta * 47.35/1.44 \
         + mByM * (-7189.233785/1.524096 + eta * 458.555/3.024 - eta * eta * 534.5/7.2))
    
def get_6PNSOCoeff(mByM):
    return PI * mByM * (1490./3. + mByM * 260.)

def get_4PNS1S2Coeff(eta):
    return 247./4.8 * eta

def get_4PNS1S2OCoeff(eta):
    return -721./4.8*eta

def get_4PNQM2SOCoeff(mByM):
    return -720./9.6 * mByM * mByM

def get_4PNSelf2SOCoeff(mByM):
    return 1./9.6 * mByM * mByM

def get_4PNQM2SCoeff(mByM):
    return 240./9.6 * mByM * mByM

def get_4PNSelf2SCoeff(mByM):
    return -7./9.6 * mByM * mByM

def get_6PNS1S2OCoeff(eta):
    return (326.75/1.12 + 557.5/1.8 * eta) * eta

def get_6PNSelf2SCoeff(mByM):
    return (-4108.25/6.72 - 108.5/1.2 * mByM + 125.5/3.6 * mByM * mByM) * mByM * mByM

def get_6PNQM2SCoeff(mByM):
    return (4703.5/8.4 + 2935./6.* mByM - 120. * mByM * mByM) * mByM * mByM

def get_10PNTidalCoeff(mByM):
    return (-288. + 264. * mByM) * mByM * mByM * mByM * mByM

def get_12PNTidalCoeff(mByM):
    return (-15895./28. + 4595./28. * mByM + 5715./14. * mByM * mByM - 325./7. * mByM * mByM * mByM) * mByM * mByM * mByM * mByM

def get_13PNTidalCoeff(mByM):
    return mByM * mByM * mByM * mByM * 24.0*(12.0 - 11.0 * mByM) * PI

def get_14PNTidalCoeff(mByM):
    mByM3 = mByM * mByM * mByM
    mByM4 = mByM3 * mByM
    return - mByM4 * 5.0 * (193986935.0/571536.0 - 14415613.0/381024.0 * mByM - 57859.0/378.0 * mByM * mByM - 209495.0/1512.0 * mByM3 + 965.0/54.0 * mByM4 - 4.0 * mByM4 * mByM)

def get_15PNTidalCoeff(mByM):
    mByM2 = mByM * mByM
    mByM3 = mByM2 * mByM
    mByM4 = mByM3 * mByM
    return mByM4 * 1.0/28.0 * PI * (27719.0 - 22415.0 * mByM + 7598.0 * mByM2 - 10520.0 * mByM3)

def get_flux_0PNCoeff(eta):
	return 32.0 * eta*eta / 5.0

def get_flux_2PNCoeff(eta):
	return -(12.47/3.36 + 3.5/1.2 * eta)

def get_flux_3PNCoeff(eta):
	return 4.0 * PI

def get_flux_4PNCoeff(eta):
	return -(44.711/9.072 - 92.71/5.04 * eta - 6.5/1.8 * eta*eta)

def get_flux_5PNCoeff(eta):
	return -(81.91/6.72 + 58.3/2.4 * eta) * PI

def get_flux_6PNCoeff(eta):
        return (664.3739519/6.9854400 + 16.0/3.0 * PI*PI - 17.12/1.05 * EulerGamma - 17.12/1.05*jnp.log(4.) + (4.1/4.8 * PI*PI - 134.543/7.776) * eta - 94.403/3.024 * eta*eta - 7.75/3.24 * eta*eta*eta)
    
def get_flux_7PNCoeff(eta):
	return -(162.85/5.04 - 214.745/1.728 * eta - 193.385/3.024 * eta*eta) * PI

def get_flux_6PNLogCoeff(eta):
	return -17.12/1.05

def get_energy_0PNCoeff(eta):
	return -eta / 2.0

def get_energy_2PNCoeff(eta):
	return -(0.75 + eta/12.0)

def get_energy_4PNCoeff(eta):
	return -(27.0/8.0 - 19.0/8.0 * eta + 1./24.0 * eta*eta)

def get_energy_6PNCoeff(eta):
	return -(67.5/6.4 - (344.45/5.76 - 20.5/9.6 * PI*PI) * eta + 15.5/9.6 * eta*eta + 3.5/518.4 * eta*eta*eta)

def get_energy_8PNCoeff(eta):
    #see arXiv:1305.4884, or eq.(26) of arXiv:1309.3474 note that in getting a4 from PRD 62, 084011 (2000), the first reference is using the fact that \omega_{static} = 0 (see arXiv:gr-qc/0105038)
    return (-39.69/1.28 + (-123.671/5.76 + 9.037/1.536 *PI*PI+ 1792./15.*jnp.log(2)+89.6/1.5*EulerGamma)* eta + (-498.449/3.456 +31.57/5.76*PI*PI)*eta*eta + 3.01/17.28 *eta*eta*eta + 7.7/3110.4*eta*eta*eta*eta)


# def get_fl

def get_PNPhasing_F2(m1, m2, S1z, S2z, lambda1, lambda2, qm_def1, qm_def2):
    """
    Implementation of XLALSimInspiralPNPhasing_F2
    """
    M = m1 + m2
    m1M = m1 / M
    m2M = m2 / M
    
    m1_s = m1 * gt
    m2_s = m2 * gt
    
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    pfaN = 3.0 / (128.0 * eta) 
    
    # Extra variables:
    S1z_2 = S1z ** 2 # spin one squared
    S2z_2 = S1z ** 2 # spin two squared
    S1z_dot_S2z = S1z * S2z # dot product spins
    
    phasing_coeffs = dict()
    phasing_log_coeffs = dict()
    
    # Get the phasing PN coefficients (LALSimInspiralPNCoefficents)
    phasing_coeffs["0PN"] = 1.0
    phasing_coeffs["1PN"] = 0.0
    phasing_coeffs["2PN"] = 5.*(74.3/8.4 + 11. * eta)/9.
    phasing_coeffs["3PN"] = -16. * PI
    phasing_coeffs["4PN"] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta*eta)/72.
    phasing_coeffs["5PN"] = 5./9.*(772.9/8.4-13.*eta) * PI
    phasing_log_coeffs["5PN"] = 5./3.*(772.9/8.4-13.*eta)*PI
    phasing_log_coeffs["6PN"] = -684.8/2.1
    phasing_coeffs["6PN"] = 11583.231236531/4.694215680 \
                        - 640./3.*PI*PI - 684.8/2.1*EulerGamma \
                            + eta*(-15737.765635/3.048192 + 225.5/1.2*PI*PI) \
                                + eta*eta*76.055/1.728 - eta*eta*eta*127.825/1.296 \
                                    + phasing_log_coeffs["6PN"] * jnp.log(4.)
    phasing_coeffs["7PN"] = PI * (770.96675/2.54016 + 378.515/1.512 * eta - 740.45/7.56 * eta * eta)
    
    # Spin
    phasing_coeffs["7PN"] += get_7PNSOCoeff(m1M) * S1z + get_7PNSOCoeff(m2M) * S2z
    phasing_coeffs["6PN"] = get_6PNSOCoeff(m1M) * S1z + get_6PNSOCoeff(m2M) * S2z \
                            + get_6PNS1S2OCoeff(eta) * S1z * S2z \
                            + (get_6PNQM2SCoeff(m1M) * qm_def1 + get_6PNSelf2SCoeff(m1M)) * S1z_2 \
                            + (get_6PNQM2SCoeff(m2M) * qm_def2 + get_6PNSelf2SCoeff(m2M)) * S2z_2
                            
    phasing_coeffs["5PN"] += get_5PNSOCoeff(m1M) * S1z + get_5PNSOCoeff(m2M) * S2z
    phasing_log_coeffs["5PN"] += 3. * (get_5PNSOCoeff(m1M) * S1z + get_5PNSOCoeff(m2M) * S2z)
    
    phasing_coeffs["4PN"] += get_4PNS1S2Coeff(eta) * S1z_dot_S2z + get_4PNS1S2OCoeff(eta) * S1z * S2z_2 \
	      + (get_4PNQM2SOCoeff(m1M) * qm_def1 + get_4PNSelf2SOCoeff(m1M)) * S1z_2 \
	      + (get_4PNQM2SOCoeff(m2M) * qm_def2 + get_4PNSelf2SOCoeff(m2M)) * S2z_2 \
	      + (get_4PNQM2SCoeff(m1M) * qm_def1 + get_4PNSelf2SCoeff(m1M)) * S1z_2 \
	      + (get_4PNQM2SCoeff(m2M) * qm_def2 + get_4PNSelf2SCoeff(m2M)) * S2z_2 
       
    phasing_coeffs["3PN"] += get_3PNSOCoeff(m1M) * S1z + get_3PNSOCoeff(m2M) * S2z
    
    # Tidal contributions
    phasing_coeffs["15PN"] = (lambda1 * get_15PNTidalCoeff(m1M) + lambda2 * get_15PNTidalCoeff(m2M))
    phasing_coeffs["14PN"] = (lambda1 * get_14PNTidalCoeff(m1M) + lambda2 * get_14PNTidalCoeff(m2M))
    phasing_coeffs["13PN"] = (lambda1 * get_13PNTidalCoeff(m1M) + lambda2 * get_13PNTidalCoeff(m2M))
    phasing_coeffs["12PN"] = (lambda1 * get_12PNTidalCoeff(m1M) + lambda2 * get_12PNTidalCoeff(m2M))
    phasing_coeffs["10PN"] = (lambda1 * get_10PNTidalCoeff(m1M) + lambda2 * get_10PNTidalCoeff(m2M))
    
    # Multiply all at the end with prefactor
    for key in phasing_coeffs.keys():
        phasing_coeffs[key] *= pfaN
    for key in phasing_log_coeffs.keys():
        phasing_log_coeffs[key] *= pfaN
    # TODO what about squares?
    
    return phasing_coeffs, phasing_log_coeffs
    


def gen_TaylorF2(f: Array, params: Array, f_ref: float):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    f_ref: Reference frequency for the waveform

    Returns:
    --------
      h0 (array): Strain
    """
    # Lets make this easier by starting in Mchirp and eta space
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    theta_intrinsic = jnp.array([m1, m2, params[2], params[3], params[4], params[5]])
    theta_extrinsic = jnp.array([params[4], params[5], params[6]])
    
    # TODO fix the quadmon properly
    quadmon1 = 0
    quadmon2 = 0

    h0 = _gen_TaylorF2(f, theta_intrinsic, theta_extrinsic, f_ref)
    return h0


def gen_TaylorF2_hphc(f: Array, params: Array, f_ref: float):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    inclination: Inclination angle of the binary [between 0 and PI]

    f_ref: Reference frequency for the waveform

    Returns:
    --------
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """
    iota = params[7]
    h0 = gen_TaylorF2(f, params, f_ref)

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc

def _gen_TaylorF2(
    f: Array,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    f_ref: float,
    phi_ref = 0
):
    """TODO finish up
    
    Note: internal units for mass are solar masses, as in LAL. 

    Args:
        f (Array): _description_
        theta_intrinsic (Array): _description_
        theta_extrinsic (Array): _description_
        coeffs (Array): _description_
        f_ref (float): _description_

    Returns:
        _type_: _description_
    """
    
    # TODO what about phi ref, what to do with that?
    
    m1, m2, chi1, chi2, lambda1, lambda2 = theta_intrinsic
    m1_s = m1 * gt
    m2_s = m2 * gt
    M = m1 + m2
    M_s = (m1 + m2) * gt
    eta = m1_s * m2_s / (M_s**2.0)
    
    piM = PI * M_s
    vISCO = 1. / jnp.sqrt(6.)
    fISCO = vISCO * vISCO * vISCO / piM
    
    # TODO is this fixed? Get qm_def parameters
    qm_def1 = get_qm_def(lambda1)
    qm_def2 = get_qm_def(lambda2)
    
    # Get the phasing coefficients
    phasing_coeffs, phasing_log_coeffs = get_PNPhasing_F2(m1, m2, chi1, chi2, lambda1, lambda2, qm_def1, qm_def2)
    
    # Get the PN order
    pfaN = 0.
    pfa1 = 0.
    pfa2 = 0.
    pfa3 = 0.
    pfa4 = 0.
    pfa5 = 0.
    pfl5 = 0.
    pfa6 = 0.
    pfl6 = 0.
    pfa7 = 0.

    # TODO what is going on here by default
    phaseO = 7
    # phaseO = XLALSimInspiralWaveformParamsLookupPNPhaseOrder(p)
    
    if phaseO == 7:
        pfa7 = phasing_coeffs["7PN"]
    elif phaseO == 6:
        pfa6 = phasing_coeffs["6PN"]
        pfl6 = phasing_log_coeffs["6PN"]
    elif phaseO == 5:
        pfa5 = phasing_coeffs["5PN"]
        pfl5 = phasing_log_coeffs["5PN"]
    elif phaseO == 4:
        pfa4 = phasing_coeffs["4PN"]
    elif phaseO == 3:
        pfa3 = phasing_coeffs["3PN"]
    elif phaseO == 2:
        pfa2 = phasing_coeffs["2PN"]
    elif phaseO == 1:
        pfa1 = phasing_coeffs["1PN"]
    elif phaseO == 0:
        pfaN = phasing_coeffs["0PN"]
    else:
        print("phaseO not recognized")
        exit()
        
    # Tidal terms:
    pft10 = 0.
    pft12 = 0.
    pft13 = 0.
    pft14 = 0.
    pft15 = 0.        
    
    # TODO what is the default here
    # tidal_order = XLALSimInspiralWaveformParamsLookupPNTidalOrder(p)
    tidal_order = "75PN"
    if tidal_order == "75PN":
        pft15 = phasing_coeffs["15PN"]
    elif tidal_order == "7PN":
        pft14 = phasing_coeffs["14PN"]
    elif tidal_order == "65PN":
        pft13 = phasing_coeffs["13PN"]
    elif tidal_order == "6PN":
        pft12 = phasing_coeffs["12PN"]
    elif tidal_order == "5PN":
        pft10 = phasing_coeffs["10PN"]
    # elif tidal_order == "0PN":
    else:
        print("Tidal order not recognized")
        exit()

    # Flux coefficients
    FTaN = get_flux_0PNCoeff(eta)
    FTa2 = get_flux_2PNCoeff(eta) # below: unused since used for SPA amplitude corrections
    FTa3 = get_flux_3PNCoeff(eta)
    FTa4 = get_flux_4PNCoeff(eta)
    FTa5 = get_flux_5PNCoeff(eta)
    FTl6 = get_flux_6PNLogCoeff(eta)
    FTa6 = get_flux_6PNCoeff(eta)
    FTa7 = get_flux_7PNCoeff(eta)
    
    # Energy coefficients
    dETaN = 2. * get_energy_0PNCoeff(eta)
    dETa1 = 2. * get_energy_2PNCoeff(eta)
    dETa2 = 3. * get_energy_4PNCoeff(eta)
    dETa3 = 4. * get_energy_6PNCoeff(eta)
    
    # TODO check if in meters or not
    r = theta_extrinsic[0] * m_per_Mpc
    amp0 = -4. * m1 * m2 / r * MRSUN * gt * jnp.sqrt(PI/12.0)
    
    ref_phasing = 0.
    if f_ref != 0:
        vref = jnp.cbrt(piM*f_ref)
        logvref = jnp.log(vref)
        
        v2ref = vref * vref
        v3ref = vref * v2ref
        v4ref = vref * v3ref
        v5ref = vref * v4ref
        v6ref = vref * v5ref
        v7ref = vref * v6ref
        v8ref = vref * v7ref
        v9ref = vref * v8ref
        v10ref = vref * v9ref
        v12ref = v2ref * v10ref
        v13ref = vref * v12ref
        v14ref = vref * v13ref
        v15ref = vref * v14ref
        ref_phasing += pfa7 * v7ref
        ref_phasing += (pfa6 + pfl6 * logvref) * v6ref
        ref_phasing += (pfa5 + pfl5 * logvref) * v5ref
        ref_phasing += pfa4 * v4ref
        ref_phasing += pfa3 * v3ref
        ref_phasing += pfa2 * v2ref
        ref_phasing += pfa1 * vref
        ref_phasing += pfaN
        
        # Tidal terms in reference phasing
        ref_phasing += pft15 * v15ref
        ref_phasing += pft14 * v14ref
        ref_phasing += pft13 * v13ref
        ref_phasing += pft12 * v12ref
        ref_phasing += pft10 * v10ref

        ref_phasing /= v5ref
    
    # TODO invent comment going here
    # f = freqs->data[i]
    v = jnp.cbrt(piM*f)
    logv = jnp.log(v)
    v2 = v * v
    v3 = v * v2
    v4 = v * v3
    v5 = v * v4
    v6 = v * v5
    v7 = v * v6
    v8 = v * v7
    v9 = v * v8
    v10 = v * v9
    v12 = v2 * v10
    v13 = v * v12
    v14 = v * v13
    v15 = v * v14
    phasing = 0.
    dEnergy = 0.
    flux = 0.

    phasing += pfa7 * v7
    phasing += (pfa6 + pfl6 * logv) * v6
    phasing += (pfa5 + pfl5 * logv) * v5
    phasing += pfa4 * v4
    phasing += pfa3 * v3
    phasing += pfa2 * v2
    phasing += pfa1 * v
    phasing += pfaN

    # Tidal terms in phasing
    phasing += pft15 * v15
    phasing += pft14 * v14
    phasing += pft13 * v13
    phasing += pft12 * v12
    phasing += pft10 * v10
    
    # TODO understand what LAL does here with amplitudeO and the comment
    flux += 1.
    dEnergy += 1.
    
    phasing /= v5
    flux *= FTaN * v10
    dEnergy *= dETaN * v
    # Note the factor of 2 b/c phi_ref is orbital phase
    
    # TODO fix this shi(f)t
    shft = 0
    phasing += shft * f - 2.*phi_ref - ref_phasing
    amp = amp0 * jnp.sqrt(-dEnergy/flux) * v
    h0 = amp * jnp.cos(phasing - PI/4) - amp * jnp.sin(phasing - PI/4) * 1.0j
    
    
    # TODO check how LAL does it, the following is from ripple
    # ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] - 2 * theta_extrinsic[2]
    # phase += ext_phase_contrib

    return h0