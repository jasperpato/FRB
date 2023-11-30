'''
Simply plots the Macquart relation.
'''

from zdm import pcosmic
import numpy as np
from zdm import parameters


def main():
	state = parameters.State()
	# set H0, Omega in state if you want, defaults to 67.66
	# state.cosmo.H0 = 67.4
	# state.cosmo.Omega_b = state.cosmo.Omega_b_h2 / (state.cosmo.H0 / 100.) ** 2
	zvals = np.linspace(0.1, 5., 50)
	macquart_relation = pcosmic.get_mean_DM(zvals, state)
	for i, z in enumerate(zvals):
		print(z, macquart_relation[i])

main()