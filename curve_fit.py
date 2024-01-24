'''
Fits a sum of exgaussians to an FRB, stores the fit information in a JSON file.
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from utils import *
import json
from globalpars import *
import matplotlib.pyplot as plt
from plot_fit import *
from calculate_data import calculate_data
import os
import globalpars
from copy import deepcopy


def raw_burst_range(xs, ys, n=globalpars.N_EFFECTIVE_WIDTHS):
	'''
	Extracts FRB from signal by taking n effective widths on either side of the peak.
	Returns low and high indices of the extracted FRB.
	'''
	eff = np.trapz(ys) / np.max(ys) # effective width
	centre = np.abs(xs).argmin()
	return max(0, int(centre - n * eff)), min(len(xs), int(centre + n * eff))


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
	bounds[0::3] = [0, MAX_A * timestep] # [-np.inf, np.inf] 
	bounds[1::3] = [xs[0], xs[-1]] 
	bounds[2::3] = [0, MAX_STDDEV * timestep]
	bounds[-1]   = [0, MAX_TIMESCALE * timestep] 

	return params, bounds.T


def rsquared(xs, ys, params):
	'''
	Returns R squared value for a set of parameters.
	'''
	residuals = ys - exgauss(xs, *params)
	ss_res = np.sum(residuals ** 2)
	ss_tot = np.sum((ys - np.mean(ys)) ** 2)
	return 1 - (ss_res / ss_tot)


def adjusted_rsquared(xs, ys, params):
	'''
	Returns adjust R squared value for a set of parameters.
	'''
	rs = rsquared(xs, ys, params)
	return 1 - (1 - rs) * (len(xs) - 1) / (len(xs) - len(params) - 1)


def update(data0, data1):
	'''
	Recursive update of data0 with data1.
	'''
	temp = deepcopy(data0['data'])
	data0.update(data1)
	data0['data'].update(temp)


def fit(xs, ys, timestep, nmin, nmax, data_file, frbname, append_data=False, visualise_for=None, stop_after=globalpars.STOP_AFTER):
	'''
	Iterates through n values and fits the sum of n exgaussians to the ys data. Saves the results to file.
	'''
	low, high = raw_burst_range(xs, ys)
	xs, ys = xs[low:high], ys[low:high]
	
	data = {
		'FRB': frbname,
		'range': [low, high],
		'data': {}
	}

	no_improvement = 0
	max_r2 = -1

	for n in range(nmin, nmax):
		try:
			print(f'N={n}')
			p0, bounds = estimate_params(n, xs, ys, timestep, visualise=visualise_for == n)
			popt, pcov = curve_fit(exgauss, xs, ys, p0, bounds=bounds)

			data['data'][str(n)] = {
				'Initial params': list(p0),
				'Params': list(popt),
				'Condition': np.linalg.cond(pcov),
				'Uncertainties': list(np.sqrt(np.diag(pcov))),
				'Adjusted R^2': (r2 := adjusted_rsquared(xs, ys, popt)),
			}

			if r2 > max_r2:
				max_r2 = r2
				no_improvement = 0
			else:
				no_improvement += 1
			
			if no_improvement == stop_after:
				print('Stopping early')
				break

		except KeyboardInterrupt: break
		except Exception as e: print(e)

	if append_data:
		with open(data_file, 'r') as f:
			data_old = json.load(f)
			update(data_old, data)
			data = data_old

	with open(data_file, 'w') as f:
		json.dump(data, f)

	return data


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--nrange', default='1,16')
	a.add_argument('--visualise-for', default=None, type=int)
	a.add_argument('--append', action='store_true')
	a.add_argument('inputs', nargs='*', default=get_files('data/pkls'))

	args = a.parse_args()

	for input in args.inputs:
		frb = os.path.basename(input)[:6]
		output = f'output/{frb}_out.json'

		if frb in []:
			print(f'Skipping {frb}')
			continue
		
		print(frb)

		frb_data = get_data(input)

		data = fit(
			frb_data.tmsarr,
			frb_data.it,
			frb_data.tresms,
			*[int(n) for n in args.nrange.split(',')],
			output,
			frb_data.frbname,
			args.append,
			args.visualise_for
		)

		calculate_data(frb_data, output)
	
		if args.visualise_for:
			plot_fitted(data.tmsarr, data.it, data.irms, output, [args.visualise_for])
	
	plt.show(block=True)
