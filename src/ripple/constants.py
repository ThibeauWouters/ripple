"""
Various constants, all in SI units.
"""

EulerGamma = 0.577215664901532860606512090082402431

MSUN = 1.988409902147041637325262574352366540e30  # kg
"""Solar mass"""

MRSUN = 1.476625038050124729627979840144936351e3
"""Geometrized nominal solar mass, m"""

# TODO explanation, taken from gwfast
GMsun_over_c3 = 4.925491025543575903411922162094833998e-6 # seconds
GMsun_over_c2 = 1.476625061404649406193430731479084713e3 # meters
uGpc = 3.085677581491367278913937957796471611e25 # meters
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc # Gpc

G = 6.67430e-11  # m^3 / kg / s^2
"""Newton's gravitational constant"""

C = 299792458.0  # m / s
"""Speed of light"""

"""Pi"""
PI = 3.141592653589793238462643383279502884

TWO_PI = 6.283185307179586476925286766559005768

gt = G * MSUN / (C ** 3.0)
"""
G MSUN / C^3 in seconds
"""

m_per_Mpc = 3.085677581491367278913937957796471611e22
"""
Meters per Mpc.
"""

clightGpc = C/3.0856778570831e+22
"""
Speed of light in vacuum (:math:`c`), in gigaparsecs per second
"""