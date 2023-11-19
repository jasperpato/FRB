'''
Fits a sum of exgaussians to an FRB.
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from utils import *
import json

# ---------- Constants ---------------------------------------------------------

STD_EXGAUSS_PEAK = 0.3

# ---------- Parameters --------------------------------------------------------

SMOOTH_STDEV = 5 # stddev for gaussian filter before estimating params

DEFAULT_STDDEV    = 1 # params for standard exGaussian before being scaled by timestep
DEFAULT_TIMESCALE = 1

A_MAX         = 5
STDDEV_MAX    = 500
TIMESCALE_MAX = 50

# ------------------------------------------------------------------------------

def estimate_params(n, xs, ys, timestep, smooth_stddev=SMOOTH_STDEV):
	'''
	Returns an array of 3*n + 1 params that are reasonable initial guesses to fit the data, and an array of bounds of the params.
	Fits N standard exGaussians at the local maxima of the burst, vertically scaled to the burst height.
	'''
	peaks, _ = find_peaks(gaussian_filter(ys, smooth_stddev))
	peak_locs = sorted(peaks, key=lambda i: ys[i], reverse=True)[:n]

	params = np.zeros(3*n+1)
	params[:-1:3] = ys[peak_locs] / STD_EXGAUSS_PEAK * timestep * DEFAULT_STDDEV
	params[1::3]  = xs[peak_locs] # centre exGaussian at the tallest peaks in ys
	params[2::3]  = DEFAULT_STDDEV * timestep
	params[-1]    = DEFAULT_TIMESCALE * timestep

	bounds = np.zeros((3*n+1, 2))
	bounds[0::3] = [0, A_MAX * timestep]         # keep vertical scale positive
	bounds[1::3] = [xs[0], xs[-1]]               # keep mean within range
	bounds[2::3] = [0, STDDEV_MAX * timestep]    # keep sd positive
	bounds[-1]   = [0, TIMESCALE_MAX * timestep] # keep ts positive

	return params, bounds.T


def fit(xs, ys, nmin, nmax, data_file):
	'''
	Iterates through n values and fits the sum of n exgaussians to the ys data. Saves the results to file.
	'''
	data = {}
	for n in range(nmin, nmax):
		d = {}
		data[str(n)] = d
		try:
			print(f'N={n}')
			p0, bounds = estimate_params(n, xs, ys, timestep)

			popt, _ = curve_fit(exgauss, xs, ys, p0, bounds=bounds)
			d['adjusted_R^2'] = adjusted_rsquared(xs, ys, popt)
			d['burst_range']  = get_bounds_model(xs, popt)
			d['params']       = list(popt)

		except Exception as e:
			if (type(e) == KeyboardInterrupt): break
			else:
				del data[str(n)]
				print(e)

	data = {
		'data': data,
		'optimum': max(data.keys(), key=lambda n: data[n]['adjusted_R^2'])
	}
	
	with open(data_file, 'w') as f:
		json.dump(data, f)

	return data


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--input', default='data/221106.pkl')
	a.add_argument('--output', default='data/data.json')
	a.add_argument('--nrange', default='1,13')

	args = a.parse_args()

	name, xs, ys, timestep, rms = get_data(args.input)
	low, high = 7600, 8500 # get_bounds_raw(ys)

	xs = xs[low:high]
	ys = ys[low:high]

	# ys[:135] = 0
	# ys[705:] = 0

	fit(xs, ys, *[int(n) for n in args.nrange.split(',')], args.output)

	# params, bounds = estimate_params(12, xs, ys, timestep)
	
	# import matplotlib.pyplot as plt
	# plt.plot(xs, ys, color='red')
	# plt.plot(xs, [exgauss(x, *params) for x in xs])
	# plt.show(block=True)
