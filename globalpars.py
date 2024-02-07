'''
Global parameters for FRB HTR data and host property correlation scripts.
JP, November 2023
'''


from collections import namedtuple


# ------------------------------------------------------------------------------

MW_DIAMETER = 30e3 # pc

#	-------------------		Curve fit constants	  ----------------------------------

STD_EXGAUSS_PEAK = 0.3

#	-------------------		Curve fit parameters   ---------------------------------

N_EFFECTIVE_WIDTHS = 20

MODEL_CENTRE_AREA = 0.95

DEFAULT_STDDEV 		= 1
DEFAULT_TIMESCALE = 1

#	-------------------   Fit plotting parameters   ------------------------------

N_WIDTHS = 1

#	-------------------   Optimal N   --------------------------------------------

R2_THRESHOLD = 0.8

NO_INCREASE_R2 = 0.1
NS_AFTER = 5

#	-------------------   DM_IGM   -----------------------------------------------

NZ = 1000
Z_MAX = 2
NDM = 5000
DM_MAX = 10000

#	------------------------ Declare FRB data type	------------------------------

frbhtr = namedtuple(
	'frbhtr', [
		'frbname','dm','nchan','ffac','tavgfac','rm','tresms','twinms','subband','refdeg','tmsarr','fmhzarr',
		'irms','qrms','urms','vrms',
		'it','qt','ut','vt','pt','lt','epts','elts','qfract','ufract','vfract','eqfract','eufract','evfract','lfract','pfract','elfract','epfract',
		'chit','psit','echit','epsit','chispec','echispec','psispec','epsispec',
		'ispec','qspec','uspec','vspec','eispec','eqspec','euspec','evspec','lspec','pspec','elspec','epspec', 
		'qfracspec','ufracspec','vfracspec','dqfracspec','dufracspec','dvfracspec','lfracspec','pfracspec','dlfracspec','dpfracspec', 
		'ids','qds','uds','vds','irmspec','qrmspec','urmspec','vrmspec'
	]
)
