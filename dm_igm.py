'''
Script to get z-DM_IGM means, medians and modes and store in a CSV file.
'''


from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
import numpy as np
from globalpars import *


def store_dms(fname, nz=NZ, zmax=Z_MAX, ndm=NDM, dmmax=DM_MAX):
	'''
	Store an array of zvals, medians, modes in a CSV file.
	'''
	# set relevant cosmological variables
	state = parameters.State()
	vparams = {}
	vparams['cosmo']={}
	vparams['cosmo']['H0'] = 67.4 # choose whatever value you wish
	vparams['cosmo']['fix_Omega_b_h2'] = True # sets Omega_b to hold Omegabh^2 constant
	vparams['IGM']={}
	vparams['IGM']['logF'] = np.log10(0.32)
	state.update_param_dict(vparams)

	# sets cosmological short-cuts based on state
	cos.set_cosmology(state)
	cos.init_dist_measures()
		
	#parameters of the grid
	# nz = 1000
	# zmax = 2
	# ndm = 5000
	# dmmax = 10000

	zmin = zmax / nz

	# get the grid of p(DM|z)
	zDMgrid, zvals, dmvals = misc_functions.get_zdm_grid(
		state,
		new=True,
		plot=False,
		method='analytic', 
		nz=nz,
		ndm=ndm,
		zmax=zmax,
		dmmax=dmmax,
		zmin=zmin
	)

	# calculate means
	weighted = zDMgrid * dmvals
	means = np.sum(weighted, axis=1) / np.sum(zDMgrid, axis=1)

	# calculate median
	cs = np.cumsum(zDMgrid, axis=1)
	cs = (cs.T / cs[:,-1]).T
	medians = np.zeros([nz])
	for i in np.arange(nz):
		imedian = np.where(cs[i,:] > 0.5)[0] # accurate to one pixel only
		medians[i] = dmvals[imedian][0]

	# calculate modes
	modes = np.zeros([nz])
	for i in np.arange(nz):
		imode = np.argmax(zDMgrid[i,:])
		modes[i] = dmvals[imode]

	with open(fname, 'w') as f:
		f.write('z,mean,median,mode\n')
		for i in range(len(zvals)):
			f.write(f'{zvals[i]},{means[i]},{medians[i]},{modes[i]}\n')


if __name__ == '__main__':
	fname = 'data/dm_igm.csv'
	store_dms(fname)