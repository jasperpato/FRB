'''
Fits the sum of N exGaussian components to an FRB signal, iterating through a range of N.
Stores the curve fit data in JSON files.

By default fits every FRB pickle file found in /data/pkls and stores the JSON files in /output.
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
import globalpars


def raw_burst_range(xs, ys, n=globalpars.N_EFFECTIVE_WIDTHS):
	'''
	Trunctates noise tails of the FRB signal by taking N effective widths on either side of the peak.
	Returns low and high indices of the extracted FRB in the centre of the signal.
	'''
	eff = np.trapz(ys) / np.max(ys) # effective width
	centre = np.abs(xs).argmin()
	return max(0, int(centre - n * eff)), min(len(xs), int(centre + n * eff))


def estimate_params(n, xs, ys, timestep, visualise=False):
	'''
	Returns an array of 3*N + 1 params that are reasonable initial guesses to fit the data, and an array of bounds of the params.
	Fits N exGaussians at the N highest peaks in the burst, vertically scaled to the peak height.
	'''
	peaks, _ = find_peaks(ys)
	num_peaks = len(peaks)

	peak_locs = sorted(peaks, key=lambda i: ys[i], reverse=True)[:n if n < num_peaks else num_peaks]
	if n > num_peaks:
		# evenly space remaining exGaussians if more than number of peaks
		peak_locs = np.append(peak_locs, np.linspace(xs[0], xs[-1], n - num_peaks, endpoint=True, dtype=int))

	params = np.zeros(3*n+1)
	params[:-1:3] = np.abs(ys[peak_locs] / STD_EXGAUSS_PEAK * timestep) # match height of ys
	params[1::3]  = xs[peak_locs]                            						# centre exGaussian at the tallest peaks in ys
	params[2::3]  = DEFAULT_STDDEV * timestep
	params[-1]    = DEFAULT_TIMESCALE * timestep

	if visualise:
		plot_single_fit(xs, ys, params)

	bounds = np.zeros((3*n+1, 2))
	bounds[0::3] = [0, np.inf] 
	bounds[1::3] = [xs[0], xs[-1]] 
	bounds[2::3] = [0, np.inf] 
	bounds[-1]   = [0, np.inf]

	return params, bounds.T


def rsquared(xs, ys, params):
	'''
	Returns R squared value for a set of fit parameters.
	'''
	residuals = ys - exgauss(xs, *params)
	ss_res = np.sum(residuals ** 2)
	ss_tot = np.sum((ys - np.mean(ys)) ** 2)
	return 1 - (ss_res / ss_tot)


def adjusted_rsquared(xs, ys, params):
	'''
	Returns adjust R squared value for a set of fit parameters.
	'''
	rs = rsquared(xs, ys, params)
	return 1 - (1 - rs) * (len(xs) - 1) / (len(xs) - len(params) - 1)


def update(data0, data1):
	'''
	Update of JSON fit data0 with data1.
	'''
	data0['data'].update(data1['data'])
	data1['data'] = data0['data']
	data0.update(data1)


def fit(xs, ys, timestep, nmin, nmax, data_file, frbname, append_data=False, visualise_for=None):
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
				'Adjusted R^2': adjusted_rsquared(xs, ys, popt),
			}

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
		frb = get_frb_name(input)
		output = f'output/{frb}_out.json'
		
		print(frb)

		frb_data = get_data(input)

		data = fit(
			frb_data.tmsarr,
			frb_data.it,
			frb_data.tresms,
			*[int(n) for n in args.nrange.split(',')],
			output,
			frb_data.frbname,
			append_data=args.append,
			visualise_for=args.visualise_for,
		)

		calculate_data(frb_data, output)
	
		if args.visualise_for:
			plot_fitted(data.tmsarr, data.it, data.irms, output, [args.visualise_for])
	
	plt.show(block=True)
