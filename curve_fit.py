'''
Fits a sum of exgaussians to an FRB.
'''


import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from utils import *
import json


def estimate_params(n, xs, ys, prev_popt=None, maxima=False):
	'''
	Returns an array of 3*n + 1 params that are reasonable initial guesses to fit the data, and an array of bounds of the params.
	Spaces out n exgaussians along range [xmin, xmax].
	'''
	params = np.zeros(3*n+1)
	params[0::3] = 0.5 # a
	params[1::3] = np.linspace(xs[0], xs[-1], n+2, endpoint=True)[1:-1] # evenly space gaussians
	# params[1::3] = xs[sorted(find_peaks(ys)[0], key=lambda i: ys[i])[:n]] # centre gaussian at the tallest peaks in ys
	params[2::3] = 1 # start the gaussians skinny
	params[-1] = 3 # ts


	if prev_popt is not None:
		prev_n = (len(prev_popt) - 1) // 3

		# replace the first exgausses with the tallest exgausses from best fitted curve in the first pass
		exgausses = sorted(np.reshape(prev_popt[:-1], (prev_n, 3)), key=lambda x: x[0], reverse=True)
		prev_params = np.ravel(exgausses)

		x = min(n, prev_n)
		params[:3*x] = prev_params[:3*x]

		params[-1] = prev_popt[-1]

	bounds = np.zeros((3*n+1, 2))
	bounds[0::3] = [0, 3] # keep vertical scale positive
	bounds[1::3] = [xs[0], xs[-1]] # mean within range
	bounds[2::3] = [0, np.inf] # sd positive
	bounds[-1] = [0, 50] # ts positive

	return params, bounds.T


def _fit(ys, nmin, nmax, data_file, prev_popt=None):
	'''
	Iterates through n values and fits the sum of n exgaussians to the ys data. Saves the results to file.
	'''
	xs = np.arange(len(ys)) - len(ys) / 2 # centre at 0 for better fitting
	data = {}
	for n in range(nmin, nmax):
		d = {}
		data[str(n)] = d
		try:
			print(f'N={n}')
			p0, bounds = estimate_params(n, xs, ys, prev_popt, maxima=True)

			try:
				popt, _ = curve_fit(sum_exgauss, xs, ys, p0, bounds=bounds)
				d['rsquared'] = rsquared(xs, ys, popt)
				d['adjusted'] = adjusted_rsquared(xs, ys, popt)
				
				popt[1:-1:3] += len(ys) / 2 # shift back the gaussian means
				d['params'] = list(popt)

			except RuntimeError as e:
				print(e)

		except KeyboardInterrupt:
			break

	with open(data_file, 'w') as f:
		json.dump(data, f)

	return data


def get_popt(data, shift):
	'''
	Returns the exgauss params with the highest adjusted r^2 value in data.
	'''
	params = np.array(max(data.values(), key=lambda d: d['adjusted'])['params'])
	params[1:-1:3] -= shift # redo the shift for the next pass
	return params


def fit(ys, fmin, fmax, smin, smax, data0, data1):
	'''
	Fits curves of different n values and stores the results to file.
	'''
	data = _fit(ys, fmin, fmax, data0) # manually set n range for first pass
	popt = get_popt(data, len(ys) / 2)
	_fit(ys, smin, smax, data1, popt) # second pass


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--frb', default='data/single_FRB_221106_I_ts_343.0_64_avg_1_200.npy')
	a.add_argument('--data0', default='data/data0.json')
	a.add_argument('--data1', default='data/data1.json')
	a.add_argument('--fmin', default=1, type=int)
	a.add_argument('--fmax', default=20, type=int)
	a.add_argument('--smin', default=1, type=int)
	a.add_argument('--smax', default=20, type=int)
	args = a.parse_args()

	ys = np.load(args.frb) # [3700:4300] # manually extract burst
	# ys[:175] = 0 # manually zero the tails to improve the fit
	# ys[450:] = 0

	fit(ys, args.fmin, args.fmax, args.smin, args.smax, args.data0, args.data1)
