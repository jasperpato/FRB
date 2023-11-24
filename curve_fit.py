'''
Fits a sum of exgaussians to an FRB.
'''


import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from utils import *
import json
from globalpars import *
import matplotlib.pyplot as plt
from confsmooth import confsmooth
from plot_fit import *

def estimate_params(n, xs, ys, timestep, visualise=False):
	'''
	Returns an array of 3*n + 1 params that are reasonable initial guesses to fit the data, and an array of bounds of the params.
	Fits N standard exGaussians at the local maxima of the burst, vertically scaled to the burst height.
	'''
	peaks, _ = find_peaks(ys)
	num_peaks = len(peaks)

	peak_locs = sorted(peaks, key=lambda i: ys[i], reverse=True)[:n if n < num_peaks else num_peaks]
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
	low, high = raw_burst_range(ys)
	xs, ys = xs[low:high], ys[low:high]

	smooth = confsmooth(ys, rms)
	
	data = {
		'range': [low, high],
		'data': {}
	}
	for n in range(nmin, nmax):
		try:
			print(f'N={n}')
			p0, bounds = estimate_params(n, xs, smooth, timestep, visualise=visualise_for == n)
			popt, pcov = curve_fit(exgauss, xs, smooth, p0, bounds=bounds)

			data['data'][str(n)] = {
				'initial_params': 			 list(p0),
				'adjusted_R^2':   			 adjusted_rsquared(xs, ys, popt),
				'burst_range':    			 (b := model_burst_range(xs, popt)),
				'burst_width':					 (b[1] - b[0]) * timestep,
				'fluence':							 sum(ys[b]),
				'condition':						 np.linalg.cond(pcov),
				'params':         			 list(popt),
				'timescale':						 popt[-1],
				'timescale_uncertainty': np.diag(pcov)[-1]
			}

		except Exception as e:
			if type(e) == KeyboardInterrupt: break
			else:
				print(e)
				del data['data'][str(n)]

	data['optimum'] = max(data['data'].keys(), key=lambda n: data['data'][n]['adjusted_R^2'])
	
	with open(data_file, 'w') as f:
		json.dump(data, f)

	return data


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--input', default='frb_data/221106.pkl')
	a.add_argument('--output', default=None)
	a.add_argument('--nrange', default='1,21')
	a.add_argument('--visualise-for', default=None, type=int)

	args = a.parse_args()

	if not args.output:
		frb = get_frb(args.input)
		args.output = f'output/{frb}_out.json'

	name, xs, ys, timestep, rms = get_data(args.input)

	data = fit(xs, ys, timestep, *[int(n) for n in args.nrange.split(',')], args.output, args.visualise_for)

	print_summary(data)
	
	if args.visualise_for:
		plot_fitted(xs, ys, rms, args.output, [args.visualise_for])
	
	plt.show(block=True)
