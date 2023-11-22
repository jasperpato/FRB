'''
Fits a sum of exgaussians to an FRB.
'''


import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from utils import *
import json
from globalpars import *
import matplotlib.pyplot as plt


def estimate_params(n, xs, ys, timestep, sigma=globalpars.PARAM_ESTIMATE_SIGMA, visualise=False):
	'''
	Returns an array of 3*n + 1 params that are reasonable initial guesses to fit the data, and an array of bounds of the params.
	Fits N standard exGaussians at the local maxima of the burst, vertically scaled to the burst height.
	'''
	smooth = gaussian_filter(ys, sigma)
	peaks, _ = find_peaks(smooth)
	num_peaks = len(peaks)

	peak_locs = sorted(peaks, key=lambda i: smooth[i], reverse=True)[:n if n < num_peaks else num_peaks]
	if n > num_peaks:
		# evenly space remaining exGaussians
		peak_locs = np.append(peak_locs, np.linspace(xs[0], xs[-1], n - num_peaks, endpoint=True, dtype=int))

	params = np.zeros(3*n+1)
	params[:-1:3] = np.abs(ys[peak_locs] / STD_EXGAUSS_PEAK * timestep) # match height of ys
	params[1::3]  = xs[peak_locs]                            						# centre exGaussian at the tallest peaks in ys
	params[2::3]  = DEFAULT_STDDEV * timestep
	params[-1]    = DEFAULT_TIMESCALE * timestep

	if visualise:
		plot_single_fit(xs, ys, params)

	bounds = np.zeros((3*n+1, 2))
	bounds[0::3] = [0, MAX_A * timestep] 
	bounds[1::3] = [xs[0], xs[-1]] 
	bounds[2::3] = [0, MAX_STDDEV * timestep]
	bounds[-1]   = [0, MAX_TIMESCALE * timestep] 

	return params, bounds.T


def fit(xs, ys, timestep, nmin, nmax, data_file, visualise_for=None):
	'''
	Iterates through n values and fits the sum of n exgaussians to the ys data. Saves the results to file.
	'''
	data = {}
	for n in range(nmin, nmax):
		d = {}
		data[str(n)] = d
		try:
			print(f'N={n}')
			p0, bounds = estimate_params(n, xs, ys, timestep, visualise=visualise_for == n)
			popt, pcov = curve_fit(exgauss, xs, ys, p0, bounds=bounds)

			d['initial_params'] 			 = list(p0)
			d['adjusted_R^2']   			 = adjusted_rsquared(xs, ys, popt)
			d['burst_range']    			 = model_burst_range(xs, popt)
			d['burst_width']					 = (d['burst_range'][1] - d['burst_range'][0]) * timestep
			d['condition']						 = np.linalg.cond(pcov)
			d['params']         			 = list(popt)
			d['timescale']						 = popt[-1]
			d['timescale_uncertainty'] = np.diag(pcov)[-1]

		except Exception as e:
			if type(e) == KeyboardInterrupt: break
			else:
				print(e)
				del d

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
	a.add_argument('--input', default='frb_data/221106.pkl')
	a.add_argument('--output', default=None)
	a.add_argument('--nrange', default='1,16')
	a.add_argument('--visualise-for', default=None, type=int)

	args = a.parse_args()

	if not args.output:
		frb = get_frb(args.input)
		args.output = f'data/{frb}_out.json'

	name, xs, ys, timestep, rms = get_data(args.input)
	low, high = raw_burst_range(ys)

	ys = ys[low:high]
	xs = xs[low:high]

	data = fit(xs, ys, timestep, *[int(n) for n in args.nrange.split(',')], args.output, args.visualise_for)

	print_summary(data)
	
	if args.visualise_for:
		from plot_fit import plot_fitted
		plot_fitted(xs, ys, rms, args.output, [args.visualise_for])
	
	plt.show(block=True)
