#
#	Functions for fitting pulse profiles
#
#								AB, 2 november 2023
#
#	Function list
#
#	expg(x, t, aa, x0, w):
#		Gaussian convolved with exponential
#
#	ncompexp(x, n, t, pars):
#		n Gaussians convolved with the same exponential	
#
#	--------------------------	Import modules	---------------------------

import os, sys
import numpy as np
from scipy import special

#	-------------------------------------------------------------------------

def expg(x, t, aa, x0, w):

#	Gaussian convolved with exponential
	
	exg		=	aa * np.exp((x0-x) / t) * special.erfc((x0 + (w*w/t) - x) / (np.sqrt(2.0)*w))
	
	return(exg)

#	-------------------------------------------------------------------------

def ncompexp(x, n, t, *pars):

#	n Gaussians convolved with the same exponential	
		
	nexg	=	0.0
	for i in range(0,n):
		nexg	=	nexg + expg(x, t, pars[0][i*3], pars[0][i*3+1], pars[0][i*3+2])
	
	return(nexg)

#	-------------------------------------------------------------------------











































