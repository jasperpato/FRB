#
#	Plotting functions
#
#								AB, October 2023
#
#	Function list
#
#			congaussfitter(plotdir,bfrbdat,xlim,fsize,ttms,ng,pol):
#					Fit profile with n Gaussians convolved with the same exponential
#
#	--------------------------	Import modules	---------------------------

import os, sys
import numpy as np
from globalpars import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mpc
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from promodels import *

plt.rc('legend', fontsize=12)    # legend fontsize
mpl.rcParams['font.size']=12
mpl.rcParams['lines.linewidth']=1.5
mpl.rcParams['axes.labelsize']=12

#frbhtr		=	namedtuple('frbhtr',['frbname','dm','nchan','ffac','tavgfac','rm','tresms','twinms','subband','refdeg','tmsarr','fmhzarr',\
		#							 'irms','qrms','urms','vrms',\	
		#							 'it','qt','ut','vt','pt','lt','epts','elts','qfract','ufract','vfract','eqfract','eufract','evfract','lfract','pfract','elfract','epfract',\
		#							 'chit','psit','echit','epsit','chispec','echispec','psispec','epsispec',\
		#							 'ispec','qspec','uspec','vspec','eispec','eqspec','euspec','evspec','lspec','pspec','elspec','epspec',\
		#							 'qfracspec','ufracspec','vfracspec','dqfracspec','dufracspec','dvfracspec','lfracspec','pfracspec','dlfracspec','dpfracspec', \
		#							 'ids','qds','uds','vds','irmspec','qrmspec','urmspec','vrmspec'])

#	----------------------------------------------------------------------------------------------------------

def congaussfitter(plotdir,bfrbdat,xlim,fsize,ttms,ng,pol):

	#	Fit profile with n Gaussians convolved with the same exponential
	
	xdat0	=	bfrbdat.tmsarr
	relind	=	np.where((xdat0 - xlim[0])*(xdat0 - xlim[1]) < 0.0)[0]
		
	if(pol=='L'):
		fdat0	=	bfrbdat.lt/np.nanmax(bfrbdat.it)
	else:
		fdat0	=	bfrbdat.it/np.nanmax(bfrbdat.it)
	
	xdat		=	xdat0[relind]
	fdat		=	fdat0[relind]
	inoise		=	bfrbdat.irms/np.nanmax(bfrbdat.it)
	
	nfitfn		=	lambda x, t, *pars : ncompexp(x, ng, t, pars)
	
	iniguess	=	np.ones(1+3*ng, dtype=float)
	
	popt,pcov	=	curve_fit(nfitfn, xdat, fdat, p0 = iniguess, bounds = (-2.0, 2.0), maxfev=50000)
	perr		=	np.sqrt(np.diag(pcov))
	
	print("Fit parameters \n")
	print(popt)
	print(perr)
	
	fig 	= plt.figure(figsize=(fsize[0], fsize[1]))
	ax 		= fig.add_axes([0.12, 0.10, 0.85,0.63])
	ax.tick_params(axis="both",direction="in",bottom=True,right=True,top=True,left=True)
	ax.text(0.8*xlim[1], 0.95, r"$\tau = %d \pm %d \: \mu$s"%(int(round(popt[0]*1.0e3)),int(round(perr[0]*1.0e3))))
	ax.axhline(c='c',ls='--')
	ax.plot(xdat, fdat, 'r-')
	ax.plot(xdat, nfitfn(xdat, *popt), 'k--')
	
	for i in range(0,ng):
		ax.plot(xdat, expg(xdat, popt[0], popt[1+3*i], popt[2+3*i], popt[3+3*i]), 'b:')
	
	#ax.set_ylim(ymax=0.19)
	ax.set_xlim(xlim)
	#ax.legend(loc='upper right')
	ax.set_ylabel(r'Normalized flux density')
	ax.xaxis.set_major_locator(ticker.MultipleLocator(ttms))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
	ax.set_xlabel(r'Time (ms)')
	ax.yaxis.set_label_coords(-0.09, 0.5)
	
	ax1 	= fig.add_axes([0.12, 0.73, 0.85,0.25])
	ax1.tick_params(axis="both",direction="in",bottom=True,right=True,top=True,left=True)
	ax1.axhline(c='c',ls='--')
	ax1.plot(xdat, (fdat - nfitfn(xdat, *popt))/inoise, 'k-')
	ax1.set_xlim(xlim)
	ax1.xaxis.set_major_locator(ticker.MultipleLocator(ttms))
	ax1.set_xticklabels([])
	ax1.set_ylabel(r'Deviation / $\sigma$')
	ax1.yaxis.set_label_coords(-0.09, 0.5)
		
	#plt.savefig("{}{}_pcomp_sb_{}_of_{}_rm_{:.2f}.pdf".format(plotdir,bfrbdat.frbname,bfrbdat.subband[0],bfrbdat.subband[1],bfrbdat.rm))
	plt.show()

	return(0)

#	----------------------------------------------------------------------------------------------------------














