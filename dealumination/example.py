import sys
sys.path.insert(0, '/users/slee75/kMC')
from zmc import Zmc
import numpy as np
r_list = []
for i in np.linspace(1,70,10):
	calc = Zmc(Ar=0.023, Ap=52.103, P_O2=i, D=20.0, rho_Cu=0.083)
	calc.calculate()
#fraction = calc.calc_frac()
	rate = calc.calc_rate()
#print(f'Fraction of Cu(II) is {fraction:.3f}')
	print(f'SCR rate is {rate:.3f} mol NO / (mol Cu * S)^-1')
	r_list.append(rate)

