from scipy.stats import exponnorm
from scipy.ndimage import gaussian_filter
import numpy as np
import pickle
import globalpars
from os.path import basename


def get_data(pkl_file):
	with open(pkl_file, 'rb') as f:
		data = pickle.load(f)
	return data # .frbname, data.tmsarr, data.it, data.tresms, data.irms


def exgauss(x, *args):
	'''
	Returns the single value f(x) of the sum of n exGaussian functions with individual
	params listed in args, followed by a single param ts.
	eg. exgauss(x, a0, u0, sd0, a1, u1, sd1, ts)
	'''
	exgauss = lambda x, a, u, sd, ts: a * exponnorm.pdf(x, ts / sd, loc=u, scale=sd)
	return sum(exgauss(x, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3))


def raw_burst_range(ys, area=globalpars.RAW_CENTRE_AREA, sigma=globalpars.RAW_SIGMA, extra_width=globalpars.RAW_EXTRA_WIDTH):
	'''
	Attempt to extract the FRB range from the raw data. Smoothens the signal and
	finds the middle range containing `area` proportion of total area under the curve.

	Finally extends the range by `extra_width` on each side.
	'''
	smooth = gaussian_filter(ys, sigma)
	total_area = np.trapz(smooth)
	low_area = (1 - area) / 2 * total_area
	high_area = (1 + area) / 2 * total_area
	i = 0
	while np.trapz(smooth[:i]) < low_area: i += 1
	low = i
	while np.trapz(smooth[:i]) < high_area: i += 1
	width = (i - low) * extra_width
	return max(0, low - width), min(len(ys), i + width)


def print_summary(data):
	'''
	Prints the relevant stats from output data.
	'''
	opt_n = data['optimum']
	rsquared = data['data'][opt_n]['adjusted_R^2']
	cond = data['data'][opt_n]['condition']
	opt_timescale = data['data'][opt_n]['timescale']
	timescale_uncertainty = data['data'][opt_n]['timescale_uncertainty']
	opt_burst_width = data['data'][opt_n]['burst_width']

	print(f'Optimum N: {opt_n}')
	print(f'Adjusted R^2: {rsquared}')
	print(f'Condition number: {cond}')
	print(f'Timescale: {opt_timescale}')
	print(f'Timescale uncertainty: {timescale_uncertainty}')
	print(f'Width: {opt_burst_width}')


def get_frb(input):
	'''
	Exract the FRB name from an input file.
	'''
	name = basename(input)
	return name[:name.index(".")]