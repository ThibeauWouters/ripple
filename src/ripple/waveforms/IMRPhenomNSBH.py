"""
Inspired by GWfast implementation, adapted for ripple, note that m1 is assumed to be the BH.

Relevant references:
    [1] `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_
    
    [2] `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_
    
    [3] `arXiv:1509.00512 <https://arxiv.org/abs/1509.00512>`_
    
    [4] `arXiv:1905.06011 <https://arxiv.org/abs/1905.06011>`_
    
TODO: NOTE: In LAL, to compute the parameter xi_tide in arXiv:1509.00512 eq. (8), the roots are extracted.
In Python this would break the possibility to vectorise so, to circumvent the issue, we compute
a grid of xi_tide as a function of the compactness, mass ratio and BH spin, and then use a 3D
interpolator. The first time the code runs, if this interpolator is not already present, it will be
computed (the base resolution of the grid is 200 pts per parameter, that we find
sufficient to reproduce LAL waveforms with good precision, given the smooth behaviour of the function,
but this can be raised if needed. In this case, it is necessary to change the name of the file assigned to self.path_xiTide_tab and the res input passed to _make_xiTide_interpolator())
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import h5py
from ..constants import gt, m_per_Mpc, PI, TWO_PI, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_quadparam_octparam, get_kappa, get_tidal_phase, get_amp0_lal
from ripple.waveforms.IMRPhenomD import get_IIb_raw_phase
from .IMRPhenomD_utils import (
    get_coeffs,
    get_delta0,
    get_delta1,
    get_delta2,
    get_delta3,
    get_delta4,
    get_transition_frequencies,
)
from .IMRPhenom_tidal_utils import get_kappa
from .IMRPhenomD_QNMdata import fM_CUT

# TODO: Check up on the interpolator, in the old GWfast code:

# self.path_xiTide_tab = os.path.join(glob.WFfilesPath, 'xiTide_Table_200.h5')
# self._make_xiTide_interpolator(res=200)
        
def Phase(f: Array, theta: Array, coeffs: Array, transition_freqs: Array) -> Array:
    """
    Compute the phase of the GW as a function of frequency, given the events parameters.
    
    # TODO: add docs
    """
    
    # TODO: remove the commented lines
    # TODO: where does f_ref come into play?
    m1, m2, chi1, chi2, lambda_ = theta
    M = m1 + m2
    eta = m1*m2/M**2
    eta2 = eta * eta # for speed
    # etaInv = 1./eta
    chi12, chi22 = chi1*chi1, chi2*chi2
    chi1dotchi2 = chi1*chi2
    
    transition_freqs = get_transition_frequencies(
        jnp.array([m1, m2, chi1, chi2]), coeffs[5], coeffs[6]
    )
    _, _, _, f4, f_RD, f_damp = transition_freqs
    
    PHI_fJoin_INS = 0.018
    fcutPar = 0.2
    
    # A non-zero tidal deformability induces a quadrupole moment (for BBH it is 1).
    # Taken from arXiv:1303.1528 eq. (54) and Tab. I
    QuadMon1, QuadMon2 = 1., jnp.where(lambda_ < 1e-5, 1., jnp.exp(0.194 + 0.0936*jnp.log(lambda_) + 0.0474*jnp.log(lambda_)*jnp.log(lambda_) - 0.00421*jnp.log(lambda_)*jnp.log(lambda_)*jnp.log(lambda_) + 0.000123*jnp.log(lambda_)*jnp.log(lambda_)*jnp.log(lambda_)*jnp.log(lambda_)))
    
    # This is needed to stabilize JAX derivatives
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    # SetaPlus1 = 1.0 + Seta
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    # q = 0.5*(1.0 + Seta - 2.0*eta)/eta
    # chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
    chi1dotchi2 = chi1*chi2
    # chi_sdotchi_a = chi_s*chi_a
    m1ByM = 0.5 * (1.0 + Seta)
    m2ByM = 0.5 * (1.0 - Seta)
    # We work in dimensionless frequency M*f, not f
    fgrid = M * gt * f
    x = (PI * fgrid) ** (2.0 / 3.0)
    
    # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
    chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
    xi = - 1.0 + chiPN
    # Compute final spin and radiated energy for IMRPhenomNSBH, the rest is equivalent to IMRPhenomD_NRTidalv2
    # Get remnant spin for assumed aligned spin system, from arXiv:1903.11622 Table I and eq. (4), (5) and (6)
    
    p1_remSp = ((-5.44187381e-03*chi1 + 7.91165608e-03) + (2.33362046e-02*chi1 + 2.47764497e-02)*eta)*eta
    p2_remSp = ((-8.56844797e-07*chi1 - 2.81727682e-06) + (6.61290966e-06*chi1 + 4.28979016e-05)*eta)*eta
    p3_remSp = ((-3.04174272e-02*chi1 + 2.54889050e-01) + (1.47549350e-01*chi1 - 4.27905832e-01)*eta)*eta
    
    modelRemSp = (1. + lambda_ * p1_remSp + lambda_*lambda_ * p2_remSp) / ((1. + lambda_*p3_remSp*p3_remSp)*(1. + lambda_*p3_remSp*p3_remSp))

    modelRemSp = jnp.where((chi1 < 0.) & (eta < 0.188), 1., modelRemSp)
    modelRemSp = jnp.where(chi1 < -0.5, 1., modelRemSp)
    modelRemSp = jnp.where(modelRemSp > 1., 1., modelRemSp)
    
    del p1_remSp, p2_remSp, p3_remSp
    
    # Work with spin variables weighted on square of the BH mass over total mass
    S1BH = chi1 * m1ByM * m1ByM
    Shat = S1BH / (m1ByM*m1ByM + m2ByM*m2ByM) # this would be = (chi1*m1*m1 + chi2*m2*m2)/(m1*m1 + m2*m2), but chi2=0 by assumption
    
    # Compute fit to L_orb in arXiv:1611.00332 eq. (16)
    Lorb = (2.*jnp.sqrt(3.)*eta + 5.24*3.8326341618708577*eta2 + 1.3*(-9.487364155598392)*eta*eta2)/(1. + 2.88*2.5134875145648374*eta) + ((-0.194)*1.0009563702914628*Shat*(4.409160174224525*eta + 0.5118334706832706*eta2 + (64. - 16.*4.409160174224525 - 4.*0.5118334706832706)*eta2*eta) + 0.0851*0.7877509372255369*Shat*Shat*(8.77367320110712*eta + (-32.060648277652994)*eta2 + (64. - 16.*8.77367320110712 - 4.*(-32.060648277652994))*eta2*eta) + 0.00954*0.6540138407185817*Shat*Shat*Shat*(22.830033250479833*eta + (-153.83722669033995)*eta2 + (64. - 16.*22.830033250479833 - 4.*(-153.83722669033995))*eta2*eta))/(1. + (-0.579)*0.8396665722805308*Shat*(1.8804718791591157 + (-4.770246856212403)*eta + 0.*eta2 + (64. - 64.*1.8804718791591157 - 16.*(-4.770246856212403) - 4.*0.)*eta2*eta)) + 0.3223660562764661*Seta*eta2*(1. + 9.332575956437443*eta)*chi1 + 2.3170397514509933*Shat*Seta*eta2*eta*(1. + (-3.2624649875884852)*eta)*chi1 + (-0.059808322561702126)*eta2*eta*chi12
    
    chif = (Lorb + S1BH)*modelRemSp
    
    # TODO: define this function below:
    # Erad = _radiatednrg(eta, chi1, chi2)
    
    # TODO: OLD GWFAST CODE: check if OK
    # Compute ringdown and damping frequencies from interpolators
    # fring = jnp.interp(chif.real, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
    # fdamp = jnp.interp(chif.real, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
    
    fring = f_RD 
    fdamp = f_damp
    
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
    
    # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
    TF2coeffs = {}
    TF2OverallAmpl = 3./(128. * eta)
    
    TF2coeffs['zero'] = 1.
    TF2coeffs['one'] = 0.
    TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
    TF2coeffs['three'] = -16.*jnp.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
    # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
    TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
    # This part is common to 5 and 5log, avoid recomputing
    TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
    TF2coeffs['five'] = (38645.*jnp.pi/756. - 65.*jnp.pi*eta/9. - TF2_5coeff_tmp)
    TF2coeffs['five_log'] = (38645.*jnp.pi/756. - 65.*jnp.pi*eta/9. - TF2_5coeff_tmp)*3.
    # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
    TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*jnp.pi*jnp.pi - 684.8/2.1*jnp.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*jnp.pi*jnp.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - jnp.log(4.)*684.8/2.1 + jnp.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + jnp.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
    TF2coeffs['six_log'] = -6848./21.
    TF2coeffs['seven'] = 77096675.*jnp.pi/254016. + 378515.*jnp.pi*eta/1512.- 74045.*jnp.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
    # Remove this part since it was not available when IMRPhenomD was tuned
    TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
    # Now translate into inspiral coefficients, label with the power in front of which they appear
    PhiInspcoeffs = {}
    
    PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
    PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(jnp.pi**(2./3.))
    PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(jnp.pi**(1./3.))
    PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(jnp.pi**(1./3.))
    PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
    PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(jnp.pi**(-1./3.))
    PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(jnp.pi**(-2./3.))
    PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/jnp.pi
    PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(jnp.pi**(-4./3.))
    PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(jnp.pi**(-5./3.))
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
    DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((jnp.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + jnp.log(jnp.pi*fInsJoin)/3.))*((jnp.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((jnp.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((jnp.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(jnp.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((jnp.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((jnp.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*jnp.pi/(3.*((jnp.pi*fInsJoin)**(8./3.)))
    DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
    # This is the first derivative of the Intermediate phase computed at fInsJoin
    DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
    
    C2Int = DPhiIns - DPhiInt
    
    # This is the inspiral phase computed at fInsJoin
    PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*jnp.log(jnp.pi*fInsJoin)/3. + PhiInspcoeffs['log']*jnp.log(jnp.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
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
    
    # TODO: this is done in the gen function?
    # fpeak = jnp.amax(fgrid, axis=0) # In LAL the maximum of the grid is used to rescale
    
    # t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
    
    # LAL sets fRef as the minimum frequency, do the same
    # TODO: where to put the f_ref?
    # fRef = M * gt * f_ref
    
    # TODO: where/when to apply the cuts?
    
    # if self.apply_fcut:
    #     phiRef = jnp.where(fRef < PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*jnp.log(jnp.pi*fRef)/3. + PhiInspcoeffs['log']*jnp.log(jnp.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, jnp.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*jnp.log(fRef))/eta + C1Int + C2Int*fRef, jnp.where(fRef < fcutPar, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * jnp.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef,0.)))

        # phis = jnp.where(fgrid < PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*jnp.log(jnp.pi*fgrid)/3. + PhiInspcoeffs['log']*jnp.log(jnp.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, jnp.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*jnp.log(fgrid))/eta + C1Int + C2Int*fgrid, jnp.where(fgrid < fcutPar, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * jnp.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid,0.)))
    # else:
        # phiRef = jnp.where(fRef < PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*jnp.log(jnp.pi*fRef)/3. + PhiInspcoeffs['log']*jnp.log(jnp.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, jnp.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*jnp.log(fRef))/eta + C1Int + C2Int*fRef, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * jnp.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef))

    # TODO: this is without the fcut, is this fine?
    phis = jnp.where(fgrid < PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*jnp.log(jnp.pi*fgrid)/3. + PhiInspcoeffs['log']*jnp.log(jnp.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, jnp.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*jnp.log(fgrid))/eta + C1Int + C2Int*fgrid, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * jnp.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid))
    
    # Add the tidal contribution to the phase
    theta_complete = jnp.array([m1, m2, chi1, chi2, 0.0, lambda_])
    tidal_phase = get_tidal_phase(x, theta_complete)
    
    phis += tidal_phase
    
    return phis
    
    ## TODO: again the fcut?
    # # This pi factor is needed to include LAL fRef rescaling, so to end up with the exact same waveform
    # if self.apply_fcut:
    #     return phis + jnp.where(fgrid < self.fcutPar, - t0*(fgrid - fRef) - phiRef + jnp.pi +  tidal_phase, 0.)
    # else:
    #     return phis - t0*(fgrid - fRef) - phiRef + jnp.pi +  tidal_phase
        
def Amp(f: Array, theta: Array, coeffs: Array, transition_freqs: Array, D: float) -> Array:
    """
    Compute the amplitude
    
    # TODO: documentation
    """
    # Usefu: remove the commented lines
    # TODO: where does f_ref come into play?
    m1, m2, chi1, chi2, lambda_ = theta
    M = m1 + m2
    eta = m1*m2/M**2
    eta2 = eta * eta # for speed
    # etaInv = 1./eta
    chi12, chi22 = chi1*chi1, chi2*chi2
    chi1dotchi2 = chi1*chi2
    
    transition_freqs = get_transition_frequencies(
        jnp.array([m1, m2, chi1, chi2]), coeffs[5], coeffs[6]
    )
    _, _, _, f4, f_RD, f_damp = transition_freqs
    
    PHI_fJoin_INS = 0.018
    fcutPar = 0.2
    
    # This is needed to stabilize JAX derivatives
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    q = 0.5*(1.0 + Seta - 2.0*eta)/eta
    SetaPlus1 = 1.0 + Seta
    # We work in dimensionless frequency M*f, not f
    fgrid = M*gt*f
    # These are m1/Mtot and m2/Mtot
    m1ByM = 0.5 * (1.0 + Seta)
    m2ByM = 0.5 * (1.0 - Seta)
    # As in arXiv:0909.2867
    chieff = m1ByM * chi1 + m2ByM * chi2
    chisum = 2.*chieff
    chiprod = chieff*chieff
    
    # compute needed IMRPhenomC attributes
    # First the SPA part, LALSimIMRPhenomC_internals.c line 38
    # Frequency-domain Amplitude coefficients
    xdotaN = 64.*eta/5.
    xdota2 = -7.43/3.36 - 11.*eta/4.
    xdota3 = 4.*jnp.pi - 11.3*chieff/1.2 + 19.*eta*chisum/6.
    xdota4 = 3.4103/1.8144 + 5*chiprod + eta*(13.661/2.016 - chiprod/8.) + 5.9*eta2/1.8
    xdota5 = -jnp.pi*(41.59/6.72 + 189.*eta/8.) - chieff*(31.571/1.008 - 116.5*eta/2.4) + chisum*(21.863*eta/1.008 - 79.*eta2/6.) - 3*chieff*chiprod/4. + 9.*eta*chieff*chiprod/4.
    xdota6 = 164.47322263/1.39708800 - 17.12*jnp.euler_gamma/1.05 + 16.*jnp.pi*jnp.pi/3 - 8.56*jnp.log(16.)/1.05 + eta*(45.1*jnp.pi*jnp.pi/4.8 - 561.98689/2.17728) + 5.41*eta2/8.96 - 5.605*eta*eta2/2.592 - 80.*jnp.pi*chieff/3. + eta*chisum*(20.*jnp.pi/3. - 113.5*chieff/3.6) + chiprod*(64.153/1.008 - 45.7*eta/3.6) - chiprod*(7.87*eta/1.44 - 30.37*eta2/1.44)
    xdota6log = -856./105.
    xdota7 = -jnp.pi*(4.415/4.032 - 358.675*eta/6.048 - 91.495*eta2/1.512) - chieff*(252.9407/2.7216 - 845.827*eta/6.048 + 415.51*eta2/8.64) + chisum*(158.0239*eta/5.4432 - 451.597*eta2/6.048 + 20.45*eta2*eta/4.32 + 107.*eta*chiprod/6. - 5.*eta2*chiprod/24.) + 12.*jnp.pi*chiprod - chiprod*chieff*(150.5/2.4 + eta/8.) + chieff*chiprod*(10.1*eta/2.4 + 3.*eta2/8.)
    # Time-domain amplitude coefficients, which also enters the fourier amplitude in this model
    AN = 8.*eta*jnp.sqrt(jnp.pi/5.)
    A2 = (-107. + 55.*eta)/42.
    A3 = 2.*jnp.pi - 4.*chieff/3. + 2.*eta*chisum/3.
    A4 = -2.173/1.512 - eta*(10.69/2.16 - 2.*chiprod) + 2.047*eta2/1.512
    A5 = -10.7*jnp.pi/2.1 + eta*(3.4*jnp.pi/2.1)
    A5imag = -24.*eta
    A6 = 270.27409/6.46800 - 8.56*jnp.euler_gamma/1.05 + 2.*jnp.pi*jnp.pi/3. + eta*(4.1*jnp.pi*jnp.pi/9.6 - 27.8185/3.3264) - 20.261*eta2/2.772 + 11.4635*eta*eta2/9.9792 - 4.28*jnp.log(16.)/1.05
    A6log = -428./105.
    A6imag = 4.28*jnp.pi/1.05
    
    z701, z702, z711, z710, z720 = 4.149e+00, -4.070e+00, -8.752e+01, -4.897e+01, 6.665e+02
    z801, z802, z811, z810, z820 = -5.472e-02, 2.094e-02, 3.554e-01, 1.151e-01, 9.640e-01
    z901, z902, z911, z910, z920 = -1.235e+00, 3.423e-01, 6.062e+00, 5.949e+00, -1.069e+01
    
    g1 = z701 * chieff + z702 * chiprod + z711 * eta * chieff + z710 * eta + z720 * eta2
    g1 = jnp.where(g1 < 0., 0., g1)
    
    del1 = z801 * chieff + z802 * chiprod + z811 * eta * chieff + z810 * eta + z820 * eta2
    del2 = z901 * chieff + z902 * chiprod + z911 * eta * chieff + z910 * eta + z920 * eta2
    del1 = jnp.where(del1 < 0., 0., del1)
    del2 = jnp.where(del2 < 1.0e-4, 1.0e-4, del2)
    
    d0 = 0.015
    
    # All the other coefficients from IMRPhenomC are not needed
    
    # Now compute NSBH coefficients
    # Get NS compactness and baryonic mass, see arXiv:1608.02582 eq. (78)
    a0Comp = 0.360
    a1Comp = -0.0355
    a2Comp = 0.000705
    
    Comp = jnp.where(lambda_ > 1., a0Comp + a1Comp*jnp.log(lambda_) + a2Comp*jnp.log(lambda_)*jnp.log(lambda_), 0.5 + (3.*a0Comp-a1Comp-1.5)*lambda_*lambda_ + (-2.*a0Comp+a1Comp+1.)*lambda_*lambda_*lambda_)
    
    # Get baryonic mass of the torus remnant of a BH-NS merger in units of the NS baryonic mass,
    # see arXiv:1509.00512 eq. (11)
    alphaTor = 0.296
    betaTor = 0.171
    # In LAL the relation is inverted each time, but this would break the vectorisation,
    # we use an interpolator on a grid of Comp, q, chi instead. Already with 100 pts per parameter the
    # agreement we find with LAL waveforms is at machine precision
    
    # TODO: this needs massive improvement!
    xiTide_interp = RegularGridInterpolator_JAX((Comps, qs, chis), xiTides, bounds_error=False)
    xiTide = xiTide_interp(jnp.asarray((jnp.asarray(Comp), jnp.asarray(q), jnp.asarray(chi1))).T)
    
    # Compute Kerr BH ISCO radius
    Z1_ISCO = 1.0 + ((1.0 - chi1*chi1)**(1./3.))*((1.0+chi1)**(1./3.) + (1.0-chi1)**(1./3.))
    Z2_ISCO = jnp.sqrt(3.0*chi1*chi1 + Z1_ISCO*Z1_ISCO)
    r_ISCO  = jnp.where(chi1>0., 3.0 + Z2_ISCO - jnp.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)), 3.0 + Z2_ISCO + jnp.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)))
    
    tmpMtorus = alphaTor * xiTide * (1.0-2.0*Comp) - betaTor * q*Comp * r_ISCO
    
    Mtorus = jnp.where(tmpMtorus>0., tmpMtorus, 0.)
    
    del tmpMtorus
    
    # Get remnant spin for assumed aligned spin system, from arXiv:1903.11622 Table I and eq. (4), (5) and (6)
    
    p1_remSp = ((-5.44187381e-03*chi1 + 7.91165608e-03) + (2.33362046e-02*chi1 + 2.47764497e-02)*eta)*eta
    p2_remSp = ((-8.56844797e-07*chi1 - 2.81727682e-06) + (6.61290966e-06*chi1 + 4.28979016e-05)*eta)*eta
    p3_remSp = ((-3.04174272e-02*chi1 + 2.54889050e-01) + (1.47549350e-01*chi1 - 4.27905832e-01)*eta)*eta
    
    modelRemSp = (1. + lambda_ * p1_remSp + lambda_*lambda_ * p2_remSp) / ((1. + lambda_*p3_remSp*p3_remSp)*(1. + lambda_*p3_remSp*p3_remSp))

    modelRemSp = jnp.where((chi1 < 0.) & (eta < 0.188), 1., modelRemSp)
    modelRemSp = jnp.where(chi1 < -0.5, 1., modelRemSp)
    modelRemSp = jnp.where(modelRemSp > 1., 1., modelRemSp)
    
    del p1_remSp, p2_remSp, p3_remSp
    
    # Work with spin variables weighted on square of the BH mass over total mass
    S1BH = chi1 * m1ByM * m1ByM
    Shat = S1BH / (m1ByM*m1ByM + m2ByM*m2ByM) # this would be = (chi1*m1*m1 + chi2*m2*m2)/(m1*m1 + m2*m2), but chi2=0 by assumption
    
    # Compute fit to L_orb in arXiv:1611.00332 eq. (16)
    Lorb = (2.*jnp.sqrt(3.)*eta + 5.24*3.8326341618708577*eta2 + 1.3*(-9.487364155598392)*eta*eta2)/(1. + 2.88*2.5134875145648374*eta) + ((-0.194)*1.0009563702914628*Shat*(4.409160174224525*eta + 0.5118334706832706*eta2 + (64. - 16.*4.409160174224525 - 4.*0.5118334706832706)*eta2*eta) + 0.0851*0.7877509372255369*Shat*Shat*(8.77367320110712*eta + (-32.060648277652994)*eta2 + (64. - 16.*8.77367320110712 - 4.*(-32.060648277652994))*eta2*eta) + 0.00954*0.6540138407185817*Shat*Shat*Shat*(22.830033250479833*eta + (-153.83722669033995)*eta2 + (64. - 16.*22.830033250479833 - 4.*(-153.83722669033995))*eta2*eta))/(1. + (-0.579)*0.8396665722805308*Shat*(1.8804718791591157 + (-4.770246856212403)*eta + 0.*eta2 + (64. - 64.*1.8804718791591157 - 16.*(-4.770246856212403) - 4.*0.)*eta2*eta)) + 0.3223660562764661*Seta*eta2*(1. + 9.332575956437443*eta)*chi1 + 2.3170397514509933*Shat*Seta*eta2*eta*(1. + (-3.2624649875884852)*eta)*chi1 + (-0.059808322561702126)*eta2*eta*chi12;
    
    chif = (Lorb + S1BH)*modelRemSp
    
    # Get remnant mass scaled to a total (initial) mass of 1
    
    p1_remM = ((-1.83417425e-03*chi1 + 2.39226041e-03) + (4.29407902e-03*chi1 + 9.79775571e-03)*eta)*eta
    p2_remM = ((2.33868869e-07*chi1 - 8.28090025e-07) + (-1.64315549e-06*chi1 + 8.08340931e-06)*eta)*eta
    p3_remM = ((-2.00726981e-02*chi1 + 1.31986011e-01) + (6.50754064e-02*chi1 - 1.42749961e-01)*eta)*eta

    modelRemM = (1. + lambda_ * p1_remM + lambda_*lambda_ * p2_remM) / ((1. + lambda_*p3_remM*p3_remM)*(1. + lambda_*p3_remM*p3_remM))
    modelRemM = jnp.where((chi1 < 0.) & (eta < 0.188), 1., modelRemM)
    modelRemM = jnp.where(chi1 < -0.5, 1., modelRemM)
    modelRemM = jnp.where(modelRemM > 1., 1., modelRemM)
    
    del p1_remM, p2_remM, p3_remM
    
    # Compute the radiated-energy fit from arXiv:1611.00332 eq. (27)
    EradNSBH = (((1. + -2.0/3.0*jnp.sqrt(2.))*eta + 0.5609904135313374*eta2 + (-0.84667563764404)*eta2*eta + 3.145145224278187*eta2*eta2)*(1. + 0.346*(-0.2091189048177395)*Shat*(1.8083565298668276 + 15.738082204419655*eta + (16. - 16.*1.8083565298668276 - 4.*15.738082204419655)*eta2) + 0.211*(-0.19709136361080587)*Shat*Shat*(4.271313308472851 + 0.*eta + (16. - 16.*4.271313308472851 - 4.*0.)*eta2) + 0.128*(-0.1588185739358418)*Shat*Shat*Shat*(31.08987570280556 + (-243.6299258830685)*eta + (16. - 16.*31.08987570280556 - 4.*(-243.6299258830685))*eta2)))/(1. + (-0.212)*2.9852925538232014*Shat*(1.5673498395263061 + (-0.5808669012986468)*eta + (16. - 16.*1.5673498395263061 - 4.*(-0.5808669012986468))*eta2)) + (-0.09803730445895877)*Seta*eta2*(1. + (-3.2283713377939134)*eta)*chi1 + (-0.01978238971523653)*Shat*Seta*eta*(1. + (-4.91667749015812)*eta)*chi1 + 0.01118530335431078*eta2*eta*chi12
    finalMass = (1.-EradNSBH)*modelRemM
    
    # Compute 22 quasi-normal mode dimensionless frequency
    kappaOm = jnp.sqrt(jnp.log(2.-chif)/jnp.log(3.))
    omega_tilde = (1.0 + kappaOm*(1.5578*jnp.exp(1j*2.9031) + 1.9510*jnp.exp(1j*5.9210)*kappaOm + 2.0997*jnp.exp(1j*2.7606)*kappaOm*kappaOm + 1.4109*jnp.exp(1j*5.9143)*kappaOm*kappaOm*kappaOm + 0.4106*jnp.exp(1j*2.7952)*(kappaOm**4)))
    
    fring = 0.5*jnp.real(omega_tilde)/jnp.pi/finalMass
    
    rtide = xiTide * (1.0 - 2.0 * Comp) / (q*Comp)
    
    q_factor = 0.5*jnp.real(omega_tilde)/jnp.imag(omega_tilde)
    
    ftide = abs(1.0/(jnp.pi*(chi1 + jnp.sqrt(rtide*rtide*rtide)))*(1.0 + 1.0 / q))
    
    # Now compute last amplitude quantities
    fring_tilde = 0.99 * 0.98 * fring
    
    gamma_correction = jnp.where(lambda_ > 1.0, 1.25, 1.0 + 0.5*lambda_ - 0.25*lambda_*lambda_)
    delta_2_prime = jnp.where(lambda_ > 1.0, 1.62496*0.25*(1. + jnp.tanh(4.0*((ftide/fring_tilde - 1.)-0.0188092)/0.338737)), del2 - 2.*(del2 - 0.81248)*lambda_ + (del2 - 0.81248)*lambda_*lambda_)
    
    sigma = delta_2_prime * fring / q_factor
    
    # Determine the type of merger we see and determine coefficients
    epsilon_tide = jnp.where(ftide < fring, 0., 2.*0.25*(1 + jnp.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.571505*Comp - 0.00508451*chi1)+0.0796251)/0.0801192)))
    
    epsilon_ins  = jnp.where(ftide < fring, jnp.where(1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*jnp.sqrt(eta) - 0.0605591*chi1)>1., 1., 1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*jnp.sqrt(eta) - 0.0605591*chi1)), jnp.where(Mtorus > 0., 1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*jnp.sqrt(eta) - 0.0605591*chi1), 1.))
    
    sigma_tide   = jnp.where(ftide < fring, jnp.where(Mtorus>0., 0.137722 - 0.293237*(Mtorus - 0.132754*Comp + 0.576669*jnp.sqrt(eta) - 0.0603749*chi1 - 0.0601185*chi1*chi1 - 0.0729134*chi1*chi1*chi1), 0.5*(0.137722 - 0.293237*(Mtorus - 0.132754*Comp + 0.576669*jnp.sqrt(eta) - 0.0603749*chi1 - 0.0601185*chi1*chi1 - 0.0729134*chi1*chi1*chi1) + 0.5*(1. - jnp.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.657424*Comp - 0.0259977*chi1)+0.206465)/0.226844)))),0.5*(1. - jnp.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.657424*Comp - 0.0259977*chi1)+0.206465)/0.226844)))
    
    f0_tilde_PN  = jnp.where(ftide < fring, jnp.where(Mtorus>0., ftide / (M*gt), ((1.0 - 1.0 / q) * fring_tilde + epsilon_ins * ftide / q)/(M*gt)), jnp.where(lambda_>1., fring_tilde/(M*gt), ((1.0 - 0.02*lambda_ + 0.01*lambda_*lambda_)*0.98*fring)/(M*gt)))
    
    f0_tilde_PM  = jnp.where(ftide < fring, jnp.where(Mtorus>0., ftide / (M*gt), ((1.0 - 1.0 / q) * fring_tilde + ftide/q)/(M*gt)), jnp.where(lambda_>1., fring_tilde/(M*gt), ((1.0 - 0.02*lambda_ + 0.01*lambda_*lambda_)*0.98*fring)/(M*gt)))
    
    f0_tilde_RD  = jnp.where(ftide < fring, 0., jnp.where(lambda_>1., fring_tilde/(M*gt), ((1.0 - 0.02*lambda_ + 0.01*lambda_*lambda_)*0.98*fring)/(M*gt)))
    
    # This can be used to output the merger type if needed
    #merger_type = ojnp.where(ftide < fring, ojnp.where(Mtorus>0., 'DISRUPTIVE', 'MILDLY_DISRUPTIVE_NO_TORUS_REMNANT'), ojnp.where(Mtorus>0.,'MILDLY_DISRUPTIVE_TORUS_REMNANT', 'NON_DISRUPTIVE'))
    
    v = (fgrid*jnp.pi)**(1./3.)

    xdot = xdotaN*(v**10)*(1. + xdota2*v*v + xdota3 * fgrid*jnp.pi + xdota4 * fgrid*jnp.pi*v + xdota5 * v*v*fgrid*jnp.pi + (xdota6 + xdota6log * 2.*jnp.log(v)) * fgrid*jnp.pi*fgrid*jnp.pi + xdota7 * v*fgrid*jnp.pi*fgrid*jnp.pi)
    ampfacTime = jnp.sqrt(abs(jnp.pi / (1.5 * v * xdot)))
    
    AmpPNre = ampfacTime * AN * v*v * (1. + A2*v*v + A3 * fgrid*jnp.pi + A4 * v*fgrid*jnp.pi + A5 * v*v*fgrid*jnp.pi + (A6 + A6log * 2.*jnp.log(v)) * fgrid*jnp.pi*fgrid*jnp.pi)
    AmpPNim = ampfacTime * AN * v*v * (A5imag * v*v*fgrid*jnp.pi + A6imag * fgrid*jnp.pi*fgrid*jnp.pi)
    
    aPN = jnp.sqrt(AmpPNre * AmpPNre + AmpPNim * AmpPNim)
    aPM = (gamma_correction * g1 * (fgrid**(5./6.)))
    
    LRD = sigma*sigma / ((fgrid - fring) * (fgrid - fring) + sigma*sigma*0.25)
    aRD = epsilon_tide * del1 * LRD * (fgrid**(-7./6.))
    
    wMinusf0_PN = 0.5 * (1. - jnp.tanh(4.*(fgrid - (epsilon_ins * f0_tilde_PN)*M*gt)/(d0 + sigma_tide)))
    wMinusf0_PM = 0.5 * (1. - jnp.tanh(4.*(fgrid - f0_tilde_PM*M*gt)/(d0 + sigma_tide)))
    wPlusf0     = 0.5 * (1. + jnp.tanh(4.*(fgrid - f0_tilde_RD*M*gt)/(d0 + sigma_tide)))
    
    amplitudeIMR = (aPN * wMinusf0_PN + aPM * wMinusf0_PM + aRD * wPlusf0)
    
    # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
    amp0 = get_amp0_lal(M, D)
    
    return amp0 * amplitudeIMR

# def _radiatednrg(self, eta, chi1, chi2):
#     """
#     Compute the total radiated energy, as in `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ eq. (3.7) and (3.8).
    
#     :param array or float eta: Symmetric mass ratio of the objects.
#     :param array or float chi1: Spin of the primary object.
#     :param array or float chi2: Spin of the secondary object.
#     :return: Total energy radiated by the system.
#     :rtype: array or float
    
#     """
#     # This is needed to stabilize JAX derivatives
#     Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
#     m1 = 0.5 * (1.0 + Seta)
#     m2 = 0.5 * (1.0 - Seta)
#     s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
    
#     EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
    
#     return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)

# def tau_star(self, f, **kwargs):
#     """
#     Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.
    
#     We use the expression in `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_ eq. (3.8b).
    
#     :param array f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
#     :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
#     :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
#     :rtype: array
    
#     """
#     Mtot_sec = kwargs['Mc']*gt/(kwargs['eta']**(3./5.))
#     v = (jnp.pi*Mtot_sec*f)**(1./3.)
#     eta = kwargs['eta']
#     eta2 = eta*eta
    
#     OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
    
#     t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*jnp.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*jnp.pi*(v**5)
#     t6  = (-10052469856691./23471078400. + 128./3.*jnp.pi*jnp.pi + 6848./105.*jnp.euler_gamma + (3147553127./3048192. - 451./12.*jnp.pi*jnp.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*jnp.log(16.*v*v))*(v**6)
#     t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*jnp.pi*(v**7)
    
#     return OverallFac*(t05 + t6 + t7)

# def fcut(self, **kwargs):
#     """
#     Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.
    
#     :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
#     :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
#     :rtype: array
    
#     """
    
#     return self.fcutPar/(kwargs['Mc']*gt/(kwargs['eta']**(3./5.)))

#################
### UTILITIES ###
#################

def _tabulate_xiTide(res=200, store=True, Compmin=.1, qmax=100., save_location: str = "./"):
    """
    Tabulate the the parameter :math:`\\xi_{\\rm tide}` in `arXiv:1509.00512 <https://arxiv.org/abs/1509.00512>`_ eq. (8) as a function of the NS compactness, the binary mass ratio and BH spin.
    
    The default ranges are chosen to cover ``LAL`` 's tuning range:
    
        - Compactness in :math:`[0.1,\, 0.5]` (``LAL`` is tuned up to :math:`\Lambda=5000`, corresponding to :math:`{\cal C}=0.109`), in *natural units*;
        - mass ratio, :math:`q=m_1/m_2`, in :math:`[1,\, 100]`;
        - chi_BH in :math:`[-1,\, 1]`.
        
    They can easily be changed if needed.
    
    :param int, optional res: Resolution of the grid in the three parameters.
    :param bool, optional store: Boolean specifying if to store or not the computed grid.
    :param float, optional Compmin: Minimum of the compactenss grid. The maximum is 0.5, corresponding to the compactness of a BH.
    :param float, optional qmax: Maximum of the mass ratio :math:`q = m_1/m_2 \geq 1`. The minimum is set to 1.
    :return: The :math:`\\xi_{\\rm tide}` tabulated grid, the used compacteness grid, mass ratio grid, and spin grid.
    :rtype: tuple(array, array, array, array)
    
    """
    Compgrid = np.linspace(Compmin, .5, res)
    qgrid = np.linspace(1., qmax, res)
    chigrid = np.linspace(-1.,1.,res)

    def sqrtxifun(Comp, q, chi):
        # Coefficients of eq. (8) of arXiv:1509.00512, using as variable sqrt(xi) (so order 10 polynomial)
        mu = q*Comp
        return np.array([1., 0., -3.*mu,  2.*chi*(mu**(3./2.)), 0., 0., -3.*q, 0., 6.*q*mu, 0., -3.*q*mu*chi*mu*chi])

    xires = np.zeros((res,res,res))
    in_time=time.time()
    for i,Comp in enumerate(Compgrid):
        for j,q in enumerate(qgrid):
            for k,chi in enumerate(chigrid):
                tmpcoeffs = sqrtxifun(Comp, q, chi)
                tmproots = np.roots(tmpcoeffs)
                # We select only real and positive solutions and take the maximum of the squares
                tmproots_rp = np.real(tmproots[(abs(np.imag(tmproots))<1e-5) & (np.real(tmproots)>0.)])
                tmpres = max(tmproots_rp*tmproots_rp)
                xires[i,j,k] = tmpres

    print('Done in %.2fs \n' %(time.time() - in_time))
    if store:
        print('Saving result...')

        # TODO: implement some checks for save location
        with h5py.File(os.path.join(save_location, 'xiTide_Table_'+str(res)+'.h5'), 'w') as out:
            out.create_dataset('Compactness', data=Compgrid, compression='gzip', shuffle=True)
            out.create_dataset('q', data=qgrid, compression='gzip', shuffle=True)
            out.create_dataset('chi', data=chigrid, compression='gzip', shuffle=True)
            out.create_dataset('xiTide', data=xires, compression='gzip', shuffle=True)
            out.attrs['npoints'] = res
            out.attrs['Compactness_min'] = Compmin
            out.attrs['q_max'] = qmax
        print('Done...')

    return xires, Compgrid, qgrid, chigrid

def _make_xiTide_interpolator(path_xiTide_tab, res=200):
    """
    Load the table of the parameter :math:`\\xi_{\\rm tide}` if present or computes it if not, and builds the needed 3-D interpolator.
    
    :param int, optional res: Resolution of the grid in compactness, mass ratio and spin.
    
    """
    if path_xiTide_tab is not None:
        if os.path.exists(path_xiTide_tab):
            print('Pre-computed xi_tide grid is present. Loading...')
            with h5py.File(path_xiTide_tab, 'r') as inp:
                Comps = jnp.array(inp['Compactness'])
                qs = jnp.array(inp['q'])
                chis = jnp.array(inp['chi'])
                xiTides = jnp.array(inp['xiTide'])
                print('Attributes of pre-computed grid: ')
                print([(k, inp.attrs[k]) for k in inp.attrs.keys()])
        else:
            print('Tabulating xi_tide...')
            xiTides, Comps, qs, chis = _tabulate_xiTide(res=res)

    else:
        print('Tabulating xi_tide...')
        xiTides, Comps, qs, chis = _tabulate_xiTide(res=res)

    xiTide_interp = RegularGridInterpolator_JAX((Comps, qs, chis), xiTides, bounds_error=False)
    
    return xiTide_interp


class RegularGridInterpolator_JAX:
    """
    Implementation of ``SciPy`` 's :py:class:`RegularGridInterpolator` in a ``JAX`` usable way. Essentially ``numpy`` in the original code is changed to ``jax.numpy`` because of assignement issues, arising when using ``vmap`` and ``jacrev``. We also changed the ``+=`` syntax which creates issues in ``JAX``.
    
    NOTE: ``bounds_error=True`` still does not work with ``vmap`` and jacrev``.
    
    """
    """
    Interpolation on a regular grid in arbitrary dimensions
    The data must be defined on a regular grid; the grid spacing however may be
    uneven. Linear and nearest-neighbor interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.
    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".
    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.
    
    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Wikipedia, "Trilinear interpolation",
           https://en.wikipedia.org/wiki/Trilinear_interpolation
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
    """
    # This class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid
    # and the original SciPy code
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

    def __init__(self, points, values, method="linear", bounds_error=False,
                 fill_value=jnp.nan):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = jnp.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not jnp.issubdtype(values.dtype, jnp.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = jnp.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    jnp.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        for i, p in enumerate(points):
            if not jnp.all(jnp.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not jnp.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        
        self.grid = tuple([jnp.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".
        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        #xi = _ndim_coords_from_arrays(xi, ndim=ndim) # Skip this checks and conversions to avoid conflicts
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not jnp.logical_and(jnp.all(self.grid[i][0] <= p),
                                      jnp.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices,
                                           norm_distances,
                                           out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances,
                                            out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result = jnp.where(out_of_bounds>0, self.fill_value, result)

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        from itertools import product
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight = weight*jnp.where(ei == i, 1 - yi, yi)
            values = values + jnp.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        print('nearest method not checked in this implementation')
        idx_res = [jnp.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = jnp.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = jnp.searchsorted(grid, x) - 1
            i = jnp.where(i < 0, 0, i)
            i = jnp.where(i > grid.size - 2, grid.size - 2, i)
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds = out_of_bounds + x < grid[0]
                out_of_bounds = out_of_bounds + x > grid[-1]
        return indices, norm_distances, out_of_bounds


### FINAL GENERATE ###

def _gen_IMRPhenomNSBH(
    f: Array,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    coeffs: Array,
    f_ref: float,
):
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * gt

    # Shift phase so that peak amplitude matches t = 0
    m1, m2, chi1, chi2, _ = theta_intrinsic
    theta_intrinsic_bbh = jnp.array([m1, m2, chi1, chi2])
    transition_freqs = get_transition_frequencies(theta_intrinsic_bbh, coeffs[5], coeffs[6])
    _, _, _, f4, f_RD, f_damp = transition_freqs
    t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta_intrinsic_bbh, coeffs, f_RD, f_damp)

    # Lets call the amplitude and phase now
    Psi = Phase(f, theta_intrinsic, coeffs, transition_freqs)
    Mf_ref = f_ref * M_s
    Psi_ref = Phase(f_ref, theta_intrinsic, coeffs, transition_freqs)
    Psi -= t0 * ((f * M_s) - Mf_ref) + Psi_ref
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] - 2 * theta_extrinsic[2]
    Psi += ext_phase_contrib
    fcut_above = lambda f: (fM_CUT / M_s)
    fcut_below = lambda f: f[jnp.abs(f - (fM_CUT / M_s)).argmin() - 1]
    fcut_true = jax.lax.cond((fM_CUT / M_s) > f[-1], fcut_above, fcut_below, f)
    # fcut_true = f[jnp.abs(f - (fM_CUT / M_s)).argmin() - 1]
    Psi = Psi * jnp.heaviside(fcut_true - f, 0.0) + 2.0 * PI * jnp.heaviside(
        f - fcut_true, 1.0
    )

    A = Amp(f, theta_intrinsic, coeffs, transition_freqs, D=theta_extrinsic[0])

    h0 = A * jnp.exp(1j * -Psi)
    return h0


def gen_IMRPhenomNSBH(f: Array, params: Array, f_ref: float):
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
    theta_intrinsic = jnp.array([m1, m2, params[2], params[3], params[4]])
    theta_extrinsic = jnp.array([params[5], params[6], params[7]])

    coeffs = get_coeffs(jnp.array([m1, m2, params[2], params[3]]))
    h0 = _gen_IMRPhenomNSBH(f, theta_intrinsic, theta_extrinsic, coeffs, f_ref)
    return h0


def gen_IMRPhenomNSBH_hphc(f: Array, params: Array, f_ref: float):
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
    iota = params[-1]
    h0 = gen_IMRPhenomNSBH(f, params, f_ref)

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
