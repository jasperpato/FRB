from scipy.stats import exponnorm
from scipy.ndimage import gaussian_filter
import numpy as np
import pickle
from globalpars import *


def get_data(pkl_file):
	'''
	Extract the relevant data from the FRB pickle file.
	'''
	with open(pkl_file, 'rb') as f:
		data = pickle.load(f)
	return data.frbname, data.tmsarr, data.it, data.tresms, data.irms


def exgauss(x, *args):
	'''
	Returns the single value f(x) of the sum of n exGaussian functions with individual
	params listed in `args`, followed by a single param ts.
	eg. exgauss(x, a0, u0, sd0, a1, u1, sd1, ...,  ts).
	'''
	exgauss = lambda x, a, u, sd, ts: a * exponnorm.pdf(x, ts / sd, loc=u, scale=sd)
	return sum(exgauss(x, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3))


def integral(x, *args):
	'''
	Integral from -inf to x of the sum of exGaussians given by params in `args`.
	'''
	single_integral = lambda x, a, u, sd, ts: a * exponnorm.cdf(x, ts / sd, loc=u, scale=sd)
	return sum(single_integral(x, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3))


def get_bounds_raw(ys, stddev=10, area=0.9, extra_width=2):
	'''
	Attempts to detect the FRB location in the raw signal. Returns suggested bounds to cut off
	the leading and trailing noise.

	Smooths the curve with a gaussian filter with `sttdev` and finds the bounds containing
	`area` under the curve.

	Then, extends the bounds by an `extra_width` on each side.
	'''
	smooth = gaussian_filter(ys, stddev)
	total_area = np.trapz(smooth)
	
	i = 0
	while np.trapz(smooth[:i]) < total_area * (1 - area) / 2:
		i += 1

	low = i

	while np.trapz(smooth[:i]) < total_area * (1 + area) / 2:
		i += 1

	width = int((i - low) * extra_width)
	return max(0, low - width), min(len(ys), i + width)


def get_bounds_model(xs, params, area=0.99):
	'''
	Get the burst range of a fitted curve by iterating through xs and finding the
	central range containing `area` proportion of the total area under the curve.
	'''
	total_area = integral(xs[-1], *params)
	i = 0
	low_area = (1 - area) / 2 * total_area
	high_area = (1 + area) / 2 * total_area

	while integral(xs[i], *params) < low_area: i += 1
	low = i
	while integral(xs[i], *params) < high_area: i += 1

	return low, i


def rsquared(xs, ys, params):
	'''
	Returns R squared value for a set of parameters.
	'''
	residuals = ys - np.array([exgauss(x, *params) for x in xs])
	ss_res = np.sum(residuals ** 2)
	ss_tot = np.sum((ys - np.mean(ys)) ** 2)
	return 1 - (ss_res / ss_tot)


def adjusted_rsquared(xs, ys, params):
	'''
	Returns adjust R squared value for a set of parameters.
	'''
	rs = rsquared(xs, ys, params)
	return 1 - (1 - rs) * (len(xs) - 1) / (len(xs) - len(params) - 1)


if __name__ == '__main__':
	name, xs, ys, timestep, rms = get_data('data/221106.pkl')
	low, high = get_bounds_raw(ys)

	print(timestep)

	xs = xs[low:high]
	ys = ys[low:high]

	import matplotlib.pyplot as plt
	plt.plot(ys, color='red')
	plt.show(block=True)