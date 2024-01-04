"""This file implements the NRTidalv2 corrections that can be applied to any BBH baseline, see http://arxiv.org/abs/1905.06011"""

import jax
import jax.numpy as jnp

from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, TWO_PI, MSUN, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms, ms_to_Mc_eta, lambda_tildes_to_lambdas, lambdas_to_lambda_tildes
import sys
from .IMRPhenomD import get_Amp0
from .utils_tidal import *

def _get_f_merger(theta: Array, physical=False):
    
    # Decompose theta
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    
    # Compute auxiliary variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1 * m2 / (m1 + m2) ** 2
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    q = 0.5*(1.0 + Seta - 2.0*eta)/eta
    Xa = 0.5 * (1.0 + Seta)
    Xb = 0.5 * (1.0 - Seta)
    kappa2T = (3.0/13.0) * ((1.0 + 12.0*Xb/Xa)*(Xa**5)*lambda1 + (1.0 + 12.0*Xa/Xb)*(Xb**5)*lambda2)
    kappa2T2 = kappa2T*kappa2T
    
    # Compute the dimensionless merger frequency (Mf) for the Planck taper filtering
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4
    
    numPT = 1.0 + n_1*kappa2T + n_2*kappa2T2
    denPT = 1.0 + d_1*kappa2T + d_2*kappa2T2
    Q_0 = a_0 / jnp.sqrt(q)
    f_merger = Q_0 * (numPT / denPT) / (2.*jnp.pi)
    
    # Convert to units
    if physical:
        f_merger = f_merger / M_s
    
    return f_merger

def get_tidal_amplitude(fgrid: Array, theta: Array, f_ref: float, distance: float):
    """Get the tidal amplitude corrections as given in equation (24) of the NRTidal paper.

    Args:
        fgrid (Array): Angular frequency, in particular, x = M f
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)
        distance (float, optional): Distance to the source in Mpc.

    Returns:
        Array: Tidal amplitude corrections A_T from NRTidalv2 paper.
    """
    
    # Global constants
    AMP_fJoin_INS = 0.014
    
    # Compute x: see NRTidalv2 paper for definition
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1 * m2 / (m1 + m2) ** 2
    eta2 = eta*eta
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    SetaPlus1 = 1.0 + Seta
    q = 0.5*(1.0 + Seta - 2.0*eta)/eta
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    Xa = 0.5 * (1.0 + Seta)
    Xb = 0.5 * (1.0 - Seta)
    kappa2T = (3.0/13.0) * ((1.0 + 12.0*Xb/Xa)*(Xa**5)*lambda1 + (1.0 + 12.0*Xa/Xb)*(Xb**5)*lambda2)
    
    # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
    chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
    xi = -1.0 + chiPN
    
    # Compute ringdown and damping frequencies from interpolators
    # TODO check this implementation
    fring, fdamp = get_fRD_fdamp(m1, m2, chi1, chi2)
    fring = M_s * fring
    fdamp = M_s * fdamp
    # fring = jnp.interp(aeff.real, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
    # fdamp = jnp.interp(aeff.real, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
    # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
    gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
    gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
    gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
    # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
    fpeak = jnp.where(gamma2 >= 1.0, jnp.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + jnp.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
    # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
    rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
    rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
    rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
    # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
    f1Interm = AMP_fJoin_INS
    f3Interm = fpeak
    dfInterm = 0.5*(f3Interm - f1Interm)
    f2Interm = f1Interm + dfInterm
    # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
    
    Acoeffs = {}
    Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(jnp.pi**(2./3.)))/672.
    Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*jnp.pi)/48.
    Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (jnp.pi**(4./3.)))/8.128512e6
    Acoeffs['five_thirds'] = ((jnp.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*jnp.pi)) / 32256.
    Acoeffs['two'] = - ((jnp.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*jnp.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*jnp.pi + 21384760320.*jnp.pi*jnp.pi)))/6.0085960704e10
    Acoeffs['seven_thirds'] = rho1
    Acoeffs['eight_thirds'] = rho2
    Acoeffs['three'] = rho3
    # v1 is the inspiral model evaluated at f1Interm
    v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
    # d1 is the derivative of the inspiral model evaluated at f1
    d1 = ((-969. + 1804.*eta)*(jnp.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*jnp.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(jnp.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(jnp.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*jnp.pi))/96768.- (f1Interm*jnp.pi*jnp.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*jnp.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*jnp.pi + 21384760320*jnp.pi*jnp.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
    # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
    v3 = jnp.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
    # d2 is the derivative of the merger-ringdown model evaluated at f3
    d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(jnp.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
    # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
    v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
    # Now some definitions to speed up
    f1  = f1Interm
    f2  = f2Interm
    f3  = f3Interm
    f12 = f1Interm*f1Interm
    f13 = f1Interm*f12
    f14 = f1Interm*f13
    f15 = f1Interm*f14
    f22 = f2Interm*f2Interm
    f23 = f2Interm*f22
    f24 = f2Interm*f23
    f32 = f3Interm*f3Interm
    f33 = f3Interm*f32
    f34 = f3Interm*f33
    f35 = f3Interm*f34
    # Finally conpute the deltas
    delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
    delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
    delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
    delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
    delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
    delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
    
    # Overall amplitude, see LALSimIMRPhenomD.c line 332
    # TODO this is broken? Why? 
    # amp0_lal = get_amp0_lal(m1+m2, distance * 1e6)
    amp0_lal = 2. * jnp.sqrt(5./(64.*jnp.pi)) * M * GMsun_over_c2_Gpc * M * GMsun_over_c3 / (distance * 1e-3)
    
    # Get IMR amplitude
    fcutPar = 0.2
    amplitudeIMR = jnp.where(fgrid < AMP_fJoin_INS, 1. + (fgrid**(2./3.))*Acoeffs['two_thirds'] + (fgrid**(4./3.)) * Acoeffs['four_thirds'] + (fgrid**(5./3.)) *  Acoeffs['five_thirds'] + (fgrid**(7./3.)) * Acoeffs['seven_thirds'] + (fgrid**(8./3.)) * Acoeffs['eight_thirds'] + fgrid * (Acoeffs['one'] + fgrid * Acoeffs['two'] + fgrid*fgrid * Acoeffs['three']), jnp.where(fgrid < fpeak, delta0 + fgrid*delta1 + fgrid*fgrid*(delta2 + fgrid*delta3 + fgrid*fgrid*delta4), jnp.where(fgrid < fcutPar,jnp.exp(-(fgrid - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((fgrid - fring)*(fgrid - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
    
    # Now compute the amplitude modification as in arXiv:1905.06011 eq. (24)
    xTidal = (jnp.pi * fgrid)**(2./3.)
    n1T    = 4.157407407407407
    n289T  = 2519.111111111111
    dTidal = 13477.8073677
    polyTidal = (1.0 + n1T*xTidal + n289T*(xTidal**(2.89)))/(1.+dTidal*(xTidal**4))
    ampTidal = -9.0*kappa2T*(xTidal**3.25)*polyTidal
    
    # Compute the dimensionless merger frequency (Mf) for the Planck taper filtering
    f_merger = _get_f_merger(theta)
    # The derivative of the Planck taper filter can return NaN in some points because of numerical issues, we declare it explicitly to avoid the issue
    @jax.custom_jvp
    def planck_taper_fun(x, y):
        # Terminate the waveform at 1.2 times the merger frequency
        a=1.2
        yp = a*y
        planck_taper = jnp.where(x < y, 1., jnp.where(x > yp, 0., 1. - 1./(jnp.exp((yp - y)/(x - y) + (yp - y)/(x - yp)) + 1.)))

        return planck_taper

    def planck_taper_fun_der(x,y):
        # Terminate the waveform at 1.2 times the merger frequency
        a=1.2
        yp = a*y
        tangent_out = jnp.where(x < y, 0., jnp.where(x > yp, 0., jnp.exp((yp - y)/(x - y) + (yp - y)/(x - yp))*((-1.+a)/(x-y) + (-1.+a)/(x-yp) + (-y+yp)/((x-y)**2) + 1.2*(-y+yp)/((x-yp)**2))/((jnp.exp((yp - y)/(x - y) + (yp - y)/(x - yp)) + 1.)**2)))
        tangent_out = jnp.nan_to_num(tangent_out)
        return tangent_out
    
    planck_taper_fun.defjvps(None, lambda y_dot, primal_out, x, y: planck_taper_fun_der(x,y) * y_dot)
    # Now compute tha Planck taper series
    # This filter causes big numerical issues at the cut when computing derivatives and the last element is very small but not 0. We fix it "by hand" with this nan_to_num which assigns 0 in place of NaN. We performed extensive checks and this does not affect any other part of the computation, only the very last point of the frequency grid in some random and rare cases.
    planck_taper = jnp.nan_to_num(planck_taper_fun(fgrid, f_merger))
    amp0 = jnp.sqrt(2.0*eta/3.0)*(jnp.pi**(-1./6.))
    
    return amp0_lal*(amp0*(fgrid**(-7./6.))*amplitudeIMR + 2*jnp.sqrt(jnp.pi/5.)*ampTidal)*planck_taper


def get_tidal_phase(fgrid: Array, theta: Array, f_ref: float) -> Array:
    """Computes the tidal phase psi_T from equation (17) of the NRTidalv2 paper.

    Args:
        fgrid (Array): Angular frequency, in particular, x = M f
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (float): Tidal parameter kappa, precomputed in the main function.

    Returns:
        Array: Tidal phase correction.
    """
    
    # Decompose theta
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    
    # Auxiliary variables
    M = m1 + m2
    M_s = M * gt
    eta = m1*m2/M**2
    eta2 = eta*eta 
    etaInv = 1./eta
    chi12, chi22 = chi1*chi1, chi2*chi2
    chi1dotchi2  = chi1*chi2
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    SetaPlus1 = 1.0 + Seta
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
    chi_sdotchi_a  = chi_s*chi_a
    q = 0.5*(1.0 + Seta - 2.0*eta)/eta
    m1ByM = 0.5 * (1.0 + Seta)
    m2ByM = 0.5 * (1.0 - Seta)
    PHI_fJoin_INS = 0.018 # IMRPhenomD parameter
    
    # Get quadrupole and octupole parameters
    quadparam1, octparam1 = get_quadparam_octparam(lambda1)
    quadparam2, octparam2 = get_quadparam_octparam(lambda2)
    
    # Subtract one from octupole moment to account for BBH baseline
    octparam1 -= 1.
    octparam2 -= 1.   
    
    # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
    chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
    xi = - 1.0 + chiPN
    
    # Compute ringdown and damping frequencies from interpolators
    # TODO check this implementation
    fring, fdamp = get_fRD_fdamp(m1, m2, chi1, chi2)
    fring = M_s * fring
    fdamp = M_s * fdamp
    
    # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
    # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
    sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
    sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
    sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
    sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
    
    # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
    # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
    beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
    beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
    beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
    
    # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
    # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
    alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
    alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
    alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
    alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
    alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
    
    # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5 PN)
    # First the nonspinning part
    TF2coeffs = {}
    TF2OverallAmpl = 3./(128. * eta)
    
    TF2coeffs['zero'] = 1.
    TF2coeffs['one'] = 0.
    TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
    TF2coeffs['three'] = -16.*PI + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
    # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
    TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*quadparam1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*quadparam2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*quadparam1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*quadparam2 - 7./9.6)*m2ByM*m2ByM*chi22
    # This part is common to 5 and 5log, avoid recomputing
    TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
    TF2coeffs['five'] = (38645.*PI/756. - 65.*PI*eta/9. - TF2_5coeff_tmp)
    TF2coeffs['five_log'] = (38645.*PI/756. - 65.*PI*eta/9. - TF2_5coeff_tmp)*3.
    # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
    TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*PI*PI - 684.8/2.1*EulerGamma + eta*(-15737.765635/3.048192 + 225.5/1.2*PI*PI) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - jnp.log(4.)*684.8/2.1 + PI*chi1*m1ByM*(1490./3. + m1ByM*260.) + PI*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*quadparam1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*quadparam2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
    TF2coeffs['six_log'] = -6848./21.
    TF2coeffs['seven'] = 77096675.*PI/254016. + 378515.*PI*eta/1512.- 74045.*PI*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
    # Remove this part since it was not available when IMRPhenomD was tuned
    TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
    
    # Now translate into inspiral coefficients, label with the power in front of which they appear
    PhiInspcoeffs = {}
    
    PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
    PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(PI**(2./3.))
    PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(PI**(1./3.))
    PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(PI**(1./3.))
    PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
    PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(PI**(-1./3.))
    PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(PI**(-2./3.))
    PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/PI
    PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(PI**(-4./3.))
    PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(PI**(-5./3.))
    PhiInspcoeffs['one'] = sigma1
    PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
    PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
    PhiInspcoeffs['two'] = sigma4 * 0.5
    
    #Now compute the coefficients to align the three parts
    
    fInsJoin = PHI_fJoin_INS
    fMRDJoin = 0.5*fring
    # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
    # Equations to solve for to get C(1) continuous join
    # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
    # Joining at fInsJoin
    # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
    # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
    # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
    DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((PI*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + jnp.log(PI*fInsJoin)/3.))*((PI*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((PI*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((PI*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(PI*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((PI*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((PI*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*PI/(3.*((PI*fInsJoin)**(8./3.)))
    DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
    # This is the first derivative of the Intermediate phase computed at fInsJoin
    DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
    
    C2Int = DPhiIns - DPhiInt
    
    # This is the inspiral phase computed at fInsJoin
    PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*jnp.log(PI*fInsJoin)/3. + PhiInspcoeffs['log']*jnp.log(PI*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
    # This is the Intermediate phase computed at fInsJoin
    PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*jnp.log(fInsJoin)
    
    C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
    
    # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
    PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*jnp.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
    DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
    DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
    PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * jnp.arctan((fMRDJoin - alpha5 * fring)/fdamp)
    
    C2MRD = DPhiIntTempVal - DPhiMRDVal
    C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
    
    # Time shift so that peak amplitude is approximately at t=0
    gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
    gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
    fpeak = jnp.where(gamma2 >= 1.0, jnp.fabs(fring - (fdamp*gamma3)/gamma2), jnp.fabs(fring + (fdamp*(-1.0 + jnp.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
    
    t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
    
    phiRef = jnp.where(f_ref < PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(f_ref**(2./3.)) + PhiInspcoeffs['third']*(f_ref**(1./3.)) + PhiInspcoeffs['third_log']*(f_ref**(1./3.))*jnp.log(PI*f_ref)/3. + PhiInspcoeffs['log']*jnp.log(PI*f_ref)/3. + PhiInspcoeffs['min_third']*(f_ref**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(f_ref**(-2./3.)) + PhiInspcoeffs['min_one']/f_ref + PhiInspcoeffs['min_four_thirds']*(f_ref**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(f_ref**(-5./3.)) + (PhiInspcoeffs['one']*f_ref + PhiInspcoeffs['four_thirds']*(f_ref**(4./3.)) + PhiInspcoeffs['five_thirds']*(f_ref**(5./3.)) + PhiInspcoeffs['two']*f_ref*f_ref)/eta, jnp.where(f_ref<fMRDJoin, (beta1*f_ref - beta3/(3.*f_ref*f_ref*f_ref) + beta2*jnp.log(f_ref))/eta + C1Int + C2Int*f_ref, (-(alpha2/f_ref) + (4.0/3.0) * (alpha3 * (f_ref**(3./4.))) + alpha1 * f_ref + alpha4 * jnp.arctan((f_ref - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*f_ref))
    phis = jnp.where(fgrid < PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*jnp.log(PI*fgrid)/3. + PhiInspcoeffs['log']*jnp.log(PI*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, jnp.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*jnp.log(fgrid))/eta + C1Int + C2Int*fgrid, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * jnp.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid))
    # Add the tidal contribution to the phase, as in arXiv:1905.06011
    # Compute the tidal coupling constant, arXiv:1905.06011 eq. (8) using Lambda = 2/3 k_2/C^5 (eq. (10))

    kappa2T = (3.0/13.0) * ((1.0 + 12.0*m2ByM/m1ByM)*(m1ByM**5)*lambda1 + (1.0 + 12.0*m1ByM/m2ByM)*(m2ByM**5)*lambda2)
    
    c_Newt   = 2.4375
    n_1      = -12.615214237993088
    n_3over2 =  19.0537346970349
    n_2      = -21.166863146081035
    n_5over2 =  90.55082156324926
    n_3      = -60.25357801943598
    d_1      = -15.11120782773667
    d_3over2 =  22.195327350624694
    d_2      =   8.064109635305156

    numTidal = 1.0 + (n_1 * ((PI*fgrid)**(2./3.))) + (n_3over2 * PI*fgrid) + (n_2 * ((PI*fgrid)**(4./3.))) + (n_5over2 * ((PI*fgrid)**(5./3.))) + (n_3 * PI*fgrid*PI*fgrid)
    denTidal = 1.0 + (d_1 * ((PI*fgrid)**(2./3.))) + (d_3over2 * PI*fgrid) + (d_2 * ((PI*fgrid)**(4./3.)))
    
    tidal_phase = - kappa2T * c_Newt / (m1ByM * m2ByM) * ((PI*fgrid)**(5./3.)) * numTidal / denTidal
    
     
    
    # Compute the higher order spin contributions
    SS_3p5PN  = - 400.*PI*(quadparam1-1.)*chi12*m1ByM*m1ByM - 400.*PI*(quadparam2-1.)*chi22*m2ByM*m2ByM
    SSS_3p5PN = 10.*((m1ByM*m1ByM+308./3.*m1ByM)*chi1+(m2ByM*m2ByM-89./3.*m2ByM)*chi2)*(quadparam1-1.)*m1ByM*m1ByM*chi12 + 10.*((m2ByM*m2ByM+308./3.*m2ByM)*chi2+(m1ByM*m1ByM-89./3.*m1ByM)*chi1)*(quadparam2-1.)*m2ByM*m2ByM*chi22 - 440.*octparam1*m1ByM*m1ByM*m1ByM*chi12*chi1 - 440.*octparam2*m2ByM*m2ByM*m2ByM*chi22*chi2
    return phis - t0*(fgrid - f_ref) - phiRef + tidal_phase + (SS_3p5PN + SSS_3p5PN)*TF2OverallAmpl*((PI*fgrid)**(2./3.))


def _gen_NRTidalv2(f: Array, theta_intrinsic: Array, theta_extrinsic: Array, f_ref: float):
    """Master internal function to get the GW strain for given parameters. The function takes 
    a BBH strain, computed from an underlying BBH approximant, e.g. IMRPhenomD, and applies the
    tidal corrections to it afterwards, according to equation (25) of the NRTidalv2 paper.

    Args:
        f (Array): Frequencies in Hz
        theta_intrinsic (Array): Internal parameters of the system: m1, m2, chi1, chi2, lambda1, lambda2
        theta_extrinsic (Array): Extrinsic parameters of the system: d_L, tc and phi_c
        h0_bbh (Array): The BBH strain of the underlying model (i.e. before applying tidal corrections).

    Returns:
        Array: Final complex-valued strain of GW.
    """

    # Compute x: see NRTidalv2 paper for definition
    m1, m2, _, _, _, _ = theta_intrinsic
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    # x = (PI * M_s * f) ** (2.0/3.0) # TODO unused?
    fgrid = M_s * f
    f_ref = M_s * f_ref
    
    amp = get_tidal_amplitude(fgrid, theta_intrinsic, f_ref, theta_extrinsic[0])
    psi_T = get_tidal_phase(fgrid, theta_intrinsic, f_ref)
    h0 = amp * jnp.exp(1.j * -psi_T)

    return h0

def gen_NRTidalv2(f: Array, params: Array, f_ref: float, IMRphenom: str, use_lambda_tildes: bool=True) -> Array:
    """
    Generate NRTidalv2 frequency domain waveform following NRTidalv2 paper.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    lambda1: Dimensionless tidal deformability of primary object
    lambda2: Dimensionless tidal deformability of secondary object
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    f_ref: Reference frequency for the waveform
    
    IMRphenom: string selecting the underlying BBH approximant
    
    use_lambda_tildes (bool): If True, signifies that given tidal parameters are lambda_tildes instead of lambda_1, lambda_2

    Returns:
    --------
        h0 (array): Strain
    """
    
    # Get component masses
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    # Internally, we use lambda_1, lambda_2 for tidal parameters, but samplers might use lambda_tildes
    if use_lambda_tildes:
        lambda1, lambda2 = lambda_tildes_to_lambdas(jnp.array([params[4], params[5], m1, m2]))
    else:
        lambda1, lambda2 = params[4], params[5]
    chi1, chi2 = params[2], params[3]
    
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    theta_extrinsic = params[6:]

    # Use BBH waveform and add tidal corrections
    return _gen_NRTidalv2(f, theta_intrinsic, theta_extrinsic, f_ref)


def gen_NRTidalv2_hphc(f: Array, params: Array, f_ref: float, IMRphenom: str="IMRPhenomD", use_lambda_tildes: bool=True):
    """
    vars array contains both intrinsic and extrinsic variables
    
    IMRphenom denotes the name of the underlying BBH approximant used, before applying tidal corrections.
    
    theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic, inclination]
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
    iota = params[-1]
    h0 = gen_NRTidalv2(f, params[:-1], f_ref, IMRphenom=IMRphenom, use_lambda_tildes=use_lambda_tildes)
    
    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
