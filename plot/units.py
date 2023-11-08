import numpy as np


s = 1
m = 1
kg = 1
pi = np.pi
h = 0.67556
M_sun = 1.989e+30 *kg
Mpc = 3.086e+22*m 
G = 6.6743e-11 *m**3* kg**-1* s**-2
c = 299792458* m / s

rho_crit = 2.77459457e11 *M_sun*h**2/Mpc**3 # critical density [M_sun h^2 / Mpc^3]
rho_crit_CLASS = rho_crit*8*pi*G/3/c**2
#print(rho_crit_CLASS*Mpc**2)

#c = 1 # |c| m = s
#rho_crit_gev = 2.77459457e11/c**2*M_sun*h**2/Mpc*8*pi/3*G
#print(rho_crit_gev)

print(100*h*1000/Mpc)