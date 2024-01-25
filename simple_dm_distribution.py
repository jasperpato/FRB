'''
Simple script to give p(DM|z) distribution.

Not computationally simplest - simplest for Clancy to make! :-)
'''

import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

import numpy as np
from matplotlib import pyplot as plt


def main():
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
	nz = 1000
	zmax = 2
	ndm = 5000
	dmmax = 10000
	
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
			
	# gets 1D array of mean values
    
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
	
	# mode
	modes = np.zeros([nz])
	for i in np.arange(nz):
		imode = np.argmax(zDMgrid[i,:])
		modes[i] = dmvals[imode]
    
	# plots things!
	plt.figure()
	
	extent = [zmin,zmax,0.,dmmax]
	aspect = zmax / dmmax * ndm/nz
	
	# normalises zDM grid to same *peak* value
	zDMgrid = (zDMgrid.T / np.max(zDMgrid, axis=1)).T
    
	plt.imshow(
		np.log10(zDMgrid.T),
		origin='lower',
		extent=extent,
		cmap=plt.get_cmap("cubehelix"),
		aspect=aspect
	)
	plt.xlabel('z')
	plt.ylabel('DM')
	cbar = plt.colorbar()
	plt.clim(-4, 0)
	cbar.set_label("$p_{\\rm cosmic}(DM|z) / p_{\\rm cosmic}^{\\rm max}(DM|z)$")
	
	plt.plot(zvals, means, label='mean')
	plt.plot(zvals, medians, label='median')
	plt.plot(zvals, modes, label='mode')
	plt.legend(loc='lower right')
	
	plt.ylim(0, zmax * 1000)
	plt.xlim(0, zmax)
	
	plt.tight_layout()
	plt.savefig('pdmgz.png')
    

if __name__ == '__main__':  
	main()