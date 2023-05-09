import math
import jax
import jax.numpy as jnp
from math import acos, atan2, sqrt, sin, cos, pi, log
from typing import Tuple
from scipy.special import factorial
from ..constants import gt, MSUN
import numpy as np
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp
from .IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
)
from ..typing import Array

#LAL_MSUN_SI = 1.9885e30  # Solar mass in kg
#LAL_MTSUN_SI = LAL_MSUN_SI * 4.925491025543575903411922162094833998e-6  # Solar mass times G over c^3 in seconds


#helper functions for LALtoPhenomP:
def ROTATEZ(angle, x, y, z):
    tmp_x = x * cos(angle) - y * sin(angle)
    tmp_y = x * sin(angle) + y * cos(angle)
    return tmp_x, tmp_y, z

def ROTATEY(angle, x, y, z):
    tmp_x = x * cos(angle) + z * sin(angle)
    tmp_z = -x * sin(angle) + z * cos(angle)
    return tmp_x, y, tmp_z

def atan2tol(y, x, tol):
    if abs(x) < tol and abs(y) < tol:
        return 0.0
    else:
        return atan2(y, x)


def LALtoPhenomP(
    m1_SI: float, m2_SI: float, f_ref: float, phiRef: float, incl: float, 
    s1x: float, s1y: float, s1z: float, s2x: float, s2y: float, s2z: float
) -> Tuple[float, float, float, float, float, float, float]:

    MAX_TOL_ATAN = 1e-10
    
    # Check arguments for sanity
    if f_ref <= 0:
        raise ValueError("Reference frequency must be positive.")
    if m1_SI <= 0:
        raise ValueError("m1 must be positive.")
    if m2_SI <= 0:
        raise ValueError("m2 must be positive.")
    if abs(s1x**2 + s1y**2 + s1z**2) > 1.0:
        raise ValueError("|S1/m1^2| must be <= 1.")
    if abs(s2x**2 + s2y**2 + s2z**2) > 1.0:
        raise ValueError("|S2/m2^2| must be <= 1.")

    m1 = m1_SI / MSUN  # Masses in solar masses
    m2 = m2_SI / MSUN
    M = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    eta = m1 * m2 / (M * M)  # Symmetric mass-ratio

    # From the components in the source frame, we can easily determine
    # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    # We also compute the spherical angles of J,
    # which we need to transform to the J frame

    # Aligned spins
    chi1_l = s1z  # Dimensionless aligned spin on BH 1
    chi2_l = s2z  # Dimensionless aligned spin on BH 2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * sqrt(s1x**2 + s1y**2)
    S2_perp = m2_2 * sqrt(s2x**2 + s2y**2)

    #print("perps: ", S1_perp, S2_perp)
    A1 = 2 + (3*m2) / (2*m1)
    A2 = 2 + (3*m1) / (2*m2)
    ASp1 = A1*S1_perp
    ASp2 = A2*S2_perp
    if (ASp2 > ASp1):
        num = ASp2
    else:
        num = ASp1
    if (m2 > m1):
        den = A2*m2_2
    else:
        den = A1*m1_2
    chip = num / den

    m_sec = M * gt
    piM = jnp.pi * m_sec
    #print("piM: ", piM)
    v_ref = (piM * f_ref)**(1/3)
    L0 = M*M * L2PNR(v_ref, eta)
    #print("L0 input: ", v_ref, eta, M)
    #print("L0: ", L0)
    # Below, _sf indicates source frame components. We will also use _Jf for J frame components
    J0x_sf = m1_2*s1x + m2_2*s2x
    J0y_sf = m1_2*s1y + m2_2*s2y
    J0z_sf = L0 + m1_2*s1z + m2_2*s2z
    J0 = jnp.sqrt(J0x_sf*J0x_sf + J0y_sf*J0y_sf + J0z_sf*J0z_sf)
  
    if J0 < 1e-10: thetaJ_sf = 0
    else: thetaJ_sf = jnp.arccos(J0z_sf / J0)

    #print(thetaJ_sf)

    if abs(J0x_sf) < MAX_TOL_ATAN and abs(J0y_sf) > MAX_TOL_ATAN:
        phiJ_sf = jnp.pi/2. - phiRef
    else:
        phiJ_sf = jnp.arctan2(J0y_sf, J0x_sf)

    phi_aligned = - phiJ_sf

    #First we determine kappa
    #in the source frame, the components of N are given in Eq (35c) of T1500606-v6
    Nx_sf = jnp.sin(incl)*jnp.cos(jnp.pi/2. - phiRef)
    Ny_sf = jnp.sin(incl)*jnp.sin(jnp.pi/2. - phiRef)
    Nz_sf = jnp.cos(incl)

    tmp_x = Nx_sf
    tmp_y = Ny_sf
    tmp_z = Nz_sf

    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

    kappa = - atan2tol(tmp_y,tmp_x, MAX_TOL_ATAN)

    #print(kappa)
    #Then we determine alpha0, by rotating LN
    tmp_x, tmp_y, tmp_z = 0,0,1
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    #print(tmp_x, tmp_y)
    if abs(tmp_x) < MAX_TOL_ATAN and abs(tmp_y) < MAX_TOL_ATAN:
        alpha0 = jnp.pi
    else:
        alpha0 = atan2(tmp_y,tmp_x)

    #Finally we determine thetaJ, by rotating N
    tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)
    Nx_Jf, Nz_Jf = tmp_x, tmp_z
    thetaJN = jnp.arccos(Nz_Jf)

    #Finally, we need to redefine the polarizations :
    #PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
    #i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ| (see (2.6) in there).
    #By contrast, the triad X,Y,N used in LAL
    #("waveframe" in the nomenclature of T1500606-v6)
    #is defined in e.g. eq (35) of this document
    #(via its components in the source frame; note we use the defautl Omega=Pi/2).
    #Both triads differ from each other by a rotation around N by an angle \zeta
    #and we need to rotate the polarizations accordingly by 2\zeta

    Xx_sf = -jnp.cos(incl)*jnp.sin(phiRef)
    Xy_sf = -jnp.cos(incl)*jnp.cos(phiRef)
    Xz_sf = jnp.sin(incl)
    tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    # Now the tmp_a are the components of X in the J frame
    # We need the polar angle of that vector in the P,Q basis of Arun et al
    # P = NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
    PArunx_Jf = 0.0
    PAruny_Jf = -1.0
    PArunz_Jf = 0.0

    # Q = NxP
    QArunx_Jf = Nz_Jf
    QAruny_Jf = 0.0
    QArunz_Jf = -Nx_Jf

    # Calculate the dot products XdotPArun and XdotQArun
    XdotPArun = tmp_x * PArunx_Jf + tmp_y * PAruny_Jf + tmp_z * PArunz_Jf
    XdotQArun = tmp_x * QArunx_Jf + tmp_y * QAruny_Jf + tmp_z * QArunz_Jf

    zeta_polariz = jnp.arctan2(XdotQArun, XdotPArun)
    return chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz



#helper functions for spin-weighted spherical harmonics:
def comb(a,b):
    temp = factorial(a)/(factorial(b) * factorial(a-b))
    return temp

def get_final_spin(m1, m2, chi1, chi2):
    "m1 m2 in solar masses"
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s ** 2.0)
    S = (chi1 * m1_s ** 2 + chi2 * m2_s ** 2) / (M_s ** 2.0)
    eta2 = eta_s * eta_s
    eta3 = eta2 * eta_s
    S2 = S * S
    S3 = S2 * S

    a = eta_s * (
        3.4641016151377544
        - 4.399247300629289 * eta_s
        + 9.397292189321194 * eta2
        - 13.180949901606242 * eta3
        + S
        * (
            (1.0 / eta_s - 0.0850917821418767 - 5.837029316602263 * eta_s)
            + (0.1014665242971878 - 2.0967746996832157 * eta_s) * S
            + (-1.3546806617824356 + 4.108962025369336 * eta_s) * S2
            + (-0.8676969352555539 + 2.064046835273906 * eta_s) * S3
        )
    )
    return a

def SpinWeightedY(theta, phi, s, l, m):
    'copied from SphericalHarmonics.c in LAL'
    if s == -2:
        if l == 2:
            if m == -2:
                fac = jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1.0 - jnp.cos(theta)) * (1.0 - jnp.cos(theta))
            elif m == -1:
                fac = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * jnp.sin(theta) * (1.0 -
                        jnp.cos(theta))
            elif m == 0:
                fac = jnp.sqrt(15.0 / (32.0 * jnp.pi)) * jnp.sin(theta) * jnp.sin(theta)
            elif m == 1:
                fac = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * jnp.sin(theta) * (1.0 + jnp.cos(theta))
            elif m == 2:
                fac = jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1.0 + jnp.cos(theta)) * (1.0 + jnp.cos(theta))
            else:
                raise ValueError(f"Invalid mode s={s}, l={l}, m={m} - require |m| <= l")
    return fac * np.exp(1j * m * phi)
    #summation = 0
    #for r in range(l-s+1):
    #    summation += (-1)**r * comb(l-s, r) * comb(l+s, r+s-m) / (np.tan(theta/2.0))**(2*r+s-m)
    #outtemp = (-1)**(l+m-s) * np.sqrt( factorial(l+m)* factorial(l-m)* (2*l+1)/(4* np.pi* factorial(l+s)* factorial#(l-s)))
    #out = outtemp * (np.sin(theta/2))**(2*l) * summation * np.exp(1j * m * phi)
    #outreal = out * np.cos(m * phi)
    #outim = out * (-np.sin(m* phi))
    #return out




def PhenomPCoreTwistUp(
    fHz, hPhenom, eta, chi1_l, chi2_l, chip, M, angcoeffs, Y2m, alphaoffset, epsilonoffset, IMRPhenomP_version):
    
    assert angcoeffs is not None
    assert Y2m is not None

    #here it is used to be LAL_MTSUN_SI 
    f = fHz * gt * M  # Frequency in geometric units

    q = (1.0 + sqrt(1.0 - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
    m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
    m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
    Sperp = chip * (m2 * m2)  # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp
    chi_eff = (m1 * chi1_l + m2 * chi2_l)  # effective spin for M=1

    if IMRPhenomP_version == 'IMRPhenomPv1_V':
        SL = chi_eff * m2  # Dimensionfull aligned spin of the largest BH. SL = m2^2 chil = m2 * M * chi_eff
    elif IMRPhenomP_version == 'IMRPhenomPv2_V' or IMRPhenomP_version == 'IMRPhenomPv2NRTidal_V':
        SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.
    else:
        raise ValueError("Unknown IMRPhenomP version! At present only v1 and v2 and tidal are available.")

    omega = pi * f
    logomega = jnp.log(omega)
    omega_cbrt = (omega)**(1/3)
    omega_cbrt2 = omega_cbrt * omega_cbrt

    alpha = (angcoeffs['alphacoeff1'] / omega
             + angcoeffs['alphacoeff2'] / omega_cbrt2
             + angcoeffs['alphacoeff3'] / omega_cbrt
             + angcoeffs['alphacoeff4'] * logomega
             + angcoeffs['alphacoeff5'] * omega_cbrt) - alphaoffset

    epsilon = (angcoeffs['epsiloncoeff1'] / omega
               + angcoeffs['epsiloncoeff2'] / omega_cbrt2
               + angcoeffs['epsiloncoeff3'] / omega_cbrt
               + angcoeffs['epsiloncoeff4'] * logomega
               + angcoeffs['epsiloncoeff5'] * omega_cbrt) - epsilonoffset

    if IMRPhenomP_version == 'IMRPhenomPv1_V':
        pass
    elif IMRPhenomP_version == 'IMRPhenomPv2_V' or IMRPhenomP_version == 'IMRPhenomPv2NRTidal_V':
        cBetah, sBetah = WignerdCoefficients(omega_cbrt, SL, eta, Sperp)
    else:
        raise ValueError("Unknown IMRPhenomP version! At present only v1 and v2 and tidal are available.")

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    d2 = [sBetah4, 2 * cBetah * sBetah3, sqrt(6) * sBetah2 * cBetah2, 2 * cBetah3 * sBetah, cBetah4]
    dm2 = [d2[4], -d2[3], d2[2], -d2[1], d2[0]]


    Y2mA = Y2m # this change means you need to pass Y2m in a 5-component list
    hp_sum = 0
    hc_sum = 0

    cexp_i_alpha = np.exp(1j * alpha)
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_im_alpha = [cexp_m2i_alpha, cexp_mi_alpha, 1.0, cexp_i_alpha, cexp_2i_alpha]
    #print("alpha:" , cexp_im_alpha)
    #print("dm2:" , dm2)
    #print("Y2m:" , Y2mA)
    

    for m in range(-2, 3):
        T2m = cexp_im_alpha[-m + 2] * dm2[m + 2] * Y2mA[m + 2]
        #print("T2m: ",T2m)
        Tm2m = cexp_im_alpha[m + 2] * d2[m + 2] * jnp.conjugate(Y2mA[m + 2])
        hp_sum += T2m + Tm2m
        #print("m=", m)
        #print(T2m, Tm2m)
        hc_sum += 1j * (T2m - Tm2m)
        #print(hc_sum)
        #print("end")

    eps_phase_hP = np.exp(-2j * epsilon) * hPhenom / 2.0
    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc

def L2PNR(v: float, eta: float) -> float:
    eta2 = eta**2
    x = v**2
    x2 = x**2
    return (eta*(1.0 + (1.5 + eta/6.0)*x + (3.375 - (19.0*eta)/8. - eta2/24.0)*x2)) / x**0.5

def WignerdCoefficients(v: float, SL: float, eta: float, Sp: float):
    
    # We define the shorthand s := Sp / (L + SL)
    L = L2PNR(v, eta)
    s = Sp / (L + SL)
    s2 = s**2
    cos_beta = 1.0 / (1.0 + s2)**0.5
    cos_beta_half = ( (1.0 + cos_beta) / 2.0 )**0.5  # cos(beta/2)
    sin_beta_half = ( (1.0 - cos_beta) / 2.0 )**0.5   # sin(beta/2)
    
    return cos_beta_half, sin_beta_half

def ComputeNNLOanglecoeffs(q, chil, chip):
    m2 = q/(1. + q)
    m1 = 1./(1. + q)
    dm = m1 - m2
    mtot = 1.
    eta = m1*m2  # mtot = 1
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    mtot2 = mtot*mtot
    mtot4 = mtot2*mtot2
    mtot6 = mtot4*mtot2
    mtot8 = mtot6*mtot2
    chil2 = chil*chil
    chip2 = chip*chip
    chip4 = chip2*chip2
    dm2 = dm*dm
    dm3 = dm2*dm
    m2_2 = m2*m2
    m2_3 = m2_2*m2
    m2_4 = m2_3*m2
    m2_5 = m2_4*m2
    m2_6 = m2_5*m2
    m2_7 = m2_6*m2
    m2_8 = m2_7*m2

    angcoeffs = {}
    angcoeffs['alphacoeff1'] = (-0.18229166666666666 - (5*dm)/(64.*m2))

    angcoeffs['alphacoeff2'] = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta))

    angcoeffs['alphacoeff3'] = (-1.7952473958333333 - (4555*dm)/(7168.*m2) -
         (15*chip2*dm*m2_3)/(128.*mtot4*eta2) -
         (35*chip2*m2_4)/(128.*mtot4*eta2) - (515*eta)/384. - (15*dm2*eta)/(256.*m2_2) -
         (175*dm*eta)/(256.*m2))

    angcoeffs['alphacoeff4'] = - (35*pi)/48. - (5*dm*pi)/(16.*m2) + \
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) + \
      (2545*m2_2*chil)/(1152.*mtot2) - \
      (5*chip2*dm*m2_5*chil)/(128.*mtot6*eta3) - \
      (35*chip2*m2_6*chil)/(384.*mtot6*eta3) + (2035*dm*m2*chil)/(21504.*mtot2*eta) + \
      (2995*m2_2*chil)/(9216.*mtot2*eta)

    angcoeffs['alphacoeff5'] = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) -
        (15*chip4*dm*m2_7)/(512.*mtot8*eta4) -
        (35*chip4*m2_8)/(512.*mtot8*eta4) -
        (485*chip2*dm*m2_3)/(14336.*mtot4*eta2) + (475*chip2*m2_4)/(6144.*mtot4*eta2) + \
        (15*chip2*dm2*m2_2)/(256.*mtot4*eta) + (145*chip2*dm*m2_3)/(512.*mtot4*eta) + \
        (575*chip2*m2_4)/(1536.*mtot4*eta) + (39695*eta)/86016. + (1615*dm2*eta)/(28672.*m2_2) - \
        (265*dm*eta)/(14336.*m2) + (955*eta2)/576. + (15*dm3*eta2)/(1024.*m2_3) + \
        (35*dm2*eta2)/(256.*m2_2) + (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*pi*chil)/(16.*mtot2*eta) - \
        (35*m2_2*pi*chil)/(16.*mtot2*eta) + (15*chip2*dm*m2_7*chil2)/(128.*mtot8*eta4) + \
        (35*chip2*m2_8*chil2)/(128.*mtot8*eta4) + \
        (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) + \
        (1645*m2_4*chil2)/(192.*mtot4*eta))
    
    angcoeffs['epsiloncoeff1'] = (-0.18229166666666666 - (5*dm)/(64.*m2))
    angcoeffs['epsiloncoeff2'] = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta))
    angcoeffs['epsiloncoeff3'] = (-1.7952473958333333 - (4555*dm)/(7168.*m2) - (515*eta)/384. -
         (15*dm2*eta)/(256.*m2_2) - (175*dm*eta)/(256.*m2))
    angcoeffs['epsiloncoeff4'] =  - (35*pi)/48. - (5*dm*pi)/(16.*m2) + \
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) + \
      (2545*m2_2*chil)/(1152.*mtot2) + (2035*dm*m2*chil)/(21504.*mtot2*eta) + \
      (2995*m2_2*chil)/(9216.*mtot2*eta)
    angcoeffs['epsiloncoeff5'] = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) + (39695*eta)/86016. +
         (1615*dm2*eta)/(28672.*m2_2) - (265*dm*eta)/(14336.*m2) + (955*eta2)/576. +
         (15*dm3*eta2)/(1024.*m2_3) + (35*dm2*eta2)/(256.*m2_2) +
         (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*pi*chil)/(16.*mtot2*eta) - (35*m2_2*pi*chil)/(16.*mtot2*eta) +
         (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
         (1645*m2_4*chil2)/(192.*mtot4*eta))
    return angcoeffs

def PhenomPOneFrequency(fs, m1, m2, chi1, chi2, phic, M):
    '''
    m1, m2: in solar masses
    phic: Orbital phase at the peak of the underlying non precessing model (rad)
    M: Total mass (Solar masses)
    '''
    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    theta_ripple = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(theta_ripple)
    transition_freqs = get_transition_frequencies(theta_ripple, coeffs[5], coeffs[6])
    phase = PhDPhase(fs, theta_ripple, coeffs, transition_freqs)
    Amp = PhDAmp(fs, theta_ripple, coeffs, transition_freqs, D=100)
    # hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple, f_ref)
    phase -= 2. * phic; # line 1316 ???
    hPhenom = Amp * (jnp.exp(1j * phase))
    phasing = -phase
    return hPhenom, phasing
    

def PhenomPcore(fs: Array, m1_SI: float, m2_SI: float, f_ref: float, phiRef: float, incl: float, s1x: float, s1y: float, s1z: float, 
                s2x: float, s2y: float, s2z: float):
    #TODO: maybe need to reverse m1 m2
    chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz = LALtoPhenomP(m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y,s2z)
    
    
    m1 = m1_SI / MSUN
    m2 = m2_SI / MSUN
    q = m2 / m1 # q>=1 
    M = m1 + m2
    chi_eff = (m1*chi1_l + m2*chi2_l) / M
    chil = (1.0+q)/q * chi_eff
    eta = m1 * m2 / (M*M)
    m_sec = M * gt
    piM = np.pi * m_sec


    omega_ref = piM * f_ref
    logomega_ref = math.log(omega_ref)
    omega_ref_cbrt = (piM * f_ref)**(1/3)  # == v0
    omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

    angcoeffs = ComputeNNLOanglecoeffs(q, chil, chip)

    alphaNNLOoffset = (angcoeffs["alphacoeff1"] / omega_ref
                   + angcoeffs["alphacoeff2"] / omega_ref_cbrt2
                   + angcoeffs["alphacoeff3"] / omega_ref_cbrt
                   + angcoeffs["alphacoeff4"] * logomega_ref
                   + angcoeffs["alphacoeff5"] * omega_ref_cbrt)

    epsilonNNLOoffset = (angcoeffs["epsiloncoeff1"] / omega_ref
                     + angcoeffs["epsiloncoeff2"] / omega_ref_cbrt2
                     + angcoeffs["epsiloncoeff3"] / omega_ref_cbrt
                     + angcoeffs["epsiloncoeff4"] * logomega_ref
                     + angcoeffs["epsiloncoeff5"] * omega_ref_cbrt)
    
    Y2m2 = SpinWeightedY(thetaJN, 0 , -2, 2, -2)
    Y2m1 = SpinWeightedY(thetaJN, 0 , -2, 2, -1)
    Y20 = SpinWeightedY(thetaJN, 0 , -2, 2, -0)
    Y21 = SpinWeightedY(thetaJN, 0 , -2, 2, 1)
    Y22 = SpinWeightedY(thetaJN, 0 , -2, 2, 2)
    Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]

    #finspin = get_final_spin(m1, m2, chi1_l, chi2_l)
    #print(finspin)


    hPhenomDs, phasings = PhenomPOneFrequency(fs, m1, m2, chi1_l, chi2_l, phiRef, M)
        
    hp, hc = PhenomPCoreTwistUp(fs, hPhenomDs, eta, chi1_l, chi2_l, chip, M, angcoeffs, 
                                Y2, alphaNNLOoffset-alpha0, epsilonNNLOoffset, "IMRPhenomPv2_V")

    return jnp.array(hp), jnp.array(hc)
    #TODO: fix the timeshift part. need to take autodiffs

