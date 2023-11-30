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
from calculate_data import calculate_data


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


def fit(xs, ys, timestep, rms, nmin, nmax, data_file, visualise_for=None):
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
				'initial_params': list(p0),
				'params': list(popt),
				'condition': np.linalg.cond(pcov),
				'uncertainties': list(np.sqrt(np.diag(pcov)))
			}

		except Exception as e:
			if type(e) == KeyboardInterrupt: break
			else:
				print(e)
				del data['data'][str(n)]

	with open(data_file, 'w') as f:
		json.dump(data, f)

	return data


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--suffix', default='_out.json')
	a.add_argument('--nrange', default='1,19')
	a.add_argument('--visualise-for', default=None, type=int)
	a.add_argument('inputs', nargs='*', default=get_data_files('data'))

	args = a.parse_args()

	for input in args.inputs:
		frb = get_frb(input)
		output = f'output/{frb}{args.suffix}'

		print(frb)

		frb_data = get_data(input)

		data = fit(
			frb_data.tmsarr,
			frb_data.it,
			frb_data.tresms,
			frb_data.irms,
			*[int(n) for n in args.nrange.split(',')],
			output,
			args.visualise_for
		)

		calculate_data(frb_data, output)
	
		if args.visualise_for:
			plot_fitted(data.tmsarr, data.it, data.irms, output, [args.visualise_for])
	
	plt.show(block=True)
