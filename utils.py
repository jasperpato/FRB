from scipy.stats import exponnorm
from scipy.ndimage import gaussian_filter
import numpy as np
import pickle
import globalpars
import os


def get_data(pkl_file):
	'''
	Return named tuple containing FRB data.
	'''
	with open(pkl_file, 'rb') as f:
		return pickle.load(f)


def exgauss(xs, *args):
	'''
	Returns the single value f(x) of the sum of n exGaussian functions with individual
	params listed in args, followed by a single param ts.
	eg. exgauss(x, a0, u0, sd0, a1, u1, sd1, ts)
	'''
	exgauss = lambda xs, a, u, sd, ts: a * exponnorm.pdf(xs, ts / sd, loc=u, scale=sd)
	return np.sum([exgauss(xs, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3)], axis=0)


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


def get_files(dir):
	'''
	Get all input files from a directory.
	'''
	return [f'{dir}/{f}' for f in os.listdir(dir)]
	

class DevNull():
	'''
	Class to write to instead of stdout to prevent output.
	'''
	def write():
		pass