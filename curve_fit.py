'''
Fits a sum of exgaussians to an FRB.
'''


import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from utils import *
import json


def estimate_params(n, xs, ys):
	'''
	Returns an array of 3*n + 1 params that are reasonable initial guesses to fit the data, and an array of bounds of the params.
	Fits N standard exGaussians at the local maxima of the burst, vertically scaled to the burst height.
	'''
	peaks, _ = find_peaks(ys)
	peak_locs = sorted(peaks, key=lambda i: ys[i], reverse=True)[:n]

	params = np.zeros(3*n+1)
	params[:-1:3] = ys[peak_locs] / 0.3 # 0.3 is the approx height of the standard exGaussian
	params[1::3] = xs[peak_locs] # centre exGaussian at the tallest peaks in ys
	params[2::3] = 1 # standard sd
	params[-1] = 1 # standard ts

	bounds = np.zeros((3*n+1, 2))
	bounds[0::3] = [0, 3] # keep vertical scale positive
	bounds[1::3] = [xs[0], xs[-1]] # mean within range
	bounds[2::3] = [0, np.inf] # sd positive
	bounds[-1] = [0, 50] # ts positive

	return params, bounds.T


def fit(ys, nmin, nmax, data_file):
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
			p0, bounds = estimate_params(n, xs, ys)

			try:
				popt, _ = curve_fit(exgauss, xs, ys, p0, bounds=bounds)
				d['adjusted_R^2'] = adjusted_rsquared(xs, ys, popt)
				
				popt[1:-1:3] += len(ys) / 2 # shift back the exGaussian means
				d['burst_range'] = burst_range(popt, len(xs))
				d['params'] = list(popt)

			except RuntimeError as e:
				print(e)

		except KeyboardInterrupt:
			break

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
	a.add_argument('--frb', default='data/single_FRB_221106_I_ts_343.0_64_avg_1_200.npy')
	a.add_argument('--data', default='data/data.json')
	a.add_argument('--nrange', default='1,16')

	args = a.parse_args()

	ys = np.load(args.frb) 
	fit(ys, *[int(n) for n in args.nrange.split(',')], args.data)
