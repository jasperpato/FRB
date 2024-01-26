'''
Calculate FRB properties for a curve fitted FRB and add properties to the JSON data file.
'''

from utils import *
import json
import numpy as np
from scipy.stats import exponnorm
import globalpars


def integral(xs, *args):
	'''
	Integral from -inf to x of the sum of exGaussians.
	'''
	single_integral = lambda xs, a, u, sd, ts: a * exponnorm.cdf(xs, ts / sd, loc=u, scale=sd)
	return np.sum([single_integral(xs, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3)], axis=0)


def ppf(xs, params, area_prop, low=None, high=None, area=None):
	'''
	Implements the ppf of the sum of exGaussians as a binary search.
	'''
	if not area:
		total_area = integral(xs[-1], *params)
		area = total_area * area_prop

	if low is None: low = 0
	if high is None: high = len(xs) - 1

	mid = (high + low) // 2

	if mid == 0 or ((a := integral(xs[mid], *params)) < area and integral(xs[mid + 1], *params) >= area):
		return mid

	if a < area:
		return ppf(xs, params, area_prop, mid + 1, high, area)
	
	else:
		return ppf(xs, params, area_prop, low, mid - 1, area)


def model_burst_range(xs, params, area_prop=globalpars.MODEL_CENTRE_AREA):
	'''
	Get the burst range of a fitted curve by finding the central range containing
	`area_prop` proportion of the total area under the curve.
	'''
	return ppf(xs, params, (1 - area_prop) / 2), ppf(xs, params, (1 + area_prop) / 2)


def replace_nan(arr):
	'''
	Replace NaN with mean of neighbours. Assumes leading and trailing zero values.
	'''
	b_val = 0
	b_i = -1

	for i in range(len(arr)):
		if not np.isnan(arr[i]):
			if i > b_i + 1:
				mean = (b_val + arr[i]) / 2
				arr[b_i+1:i] = mean
			b_i = i
			b_val = arr[i]

	if np.isnan(arr[-1]):
		arr[b_i+1:] = b_val / 2

	return arr


def polarisation_fluence(arr, error):
	'''
	Returns polarisation fluence and associated error.
	'''
	return [
		np.sum(arr),
		np.sum(error ** 2) ** 0.5
	]


def polarisation_fraction(pol, error, fluence, fluence_error):
	'''
	Returns polarisation fraction of intensity fluence and associated error.
	'''
	return [
		(f := pol / fluence),
		f * np.hypot(error / pol, fluence_error / fluence)
	]


def get_threshold_r2(data, threshold=globalpars.R2_THRESHOLD):
	'''
	Find the minimum N that exceeds the threshold R^2 value.
	If none exceed the threshold, return None.
	'''
	for n, d in sorted(data.items(), key=lambda t: t[0]):
		if d['Adjusted R^2'] >= threshold:
			return n


def calculate_data(frb_data, data_file):
	'''
	Calculate FRB properties for a curve fitted FRB. Add the properties to the
	JSON file.
	'''
	print(f'Calculating data for {frb_data.frbname}')

	with open(data_file, 'r') as f:
		data = json.load(f)

	xs = frb_data.tmsarr
	it = frb_data.it
	timestep = frb_data.tresms

	lt = replace_nan(frb_data.lt)
	elts = replace_nan(frb_data.elts)
	pt = replace_nan(frb_data.pt)
	epts = replace_nan(frb_data.epts)

	for n, d in data['data'].items():
		d['Burst range'] = (b := model_burst_range(xs, d['Params']))
		d['Burst width (ms)'] = (b[1] - b[0]) * timestep

		# use model bounds
		it_bounded = it[b[0]:b[1]]
		lt_bounded = lt[b[0]:b[1]]
		pt_bounded = pt[b[0]:b[1]]
		elts_bounded = elts[b[0]:b[1]]
		epts_bounded = epts[b[0]:b[1]]
		
		d['Fluence'] = (f := [np.sum(it_bounded), frb_data.irms * len(xs) ** 0.5]) # fluence and associated error
		d['Linear polarisation fluence'] = (l := polarisation_fluence(lt_bounded, elts_bounded))
		d['Total polarisation fluence'] = (p := polarisation_fluence(pt_bounded, epts_bounded))
		
		d['Linear polarisation fraction'] = polarisation_fraction(*l, *f)
		d['Total polarisation fraction'] = polarisation_fraction(*p, *f)
		
		d['Tau_obs (ms)'] = [d['Params'][-1], d['Uncertainties'][-1]] # timescale and timescale error
		
		data['data'][n] = dict(sorted(d.items()))

	data['Max R^2'] = max(data['data'].keys(), key=lambda n: data['data'][n]['Adjusted R^2'])
	data['Threshold R^2'] = get_threshold_r2(data['data']) or data['Max R^2']
	
	with open(data_file, 'w') as f:
		json.dump(data, f)


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('inputs', nargs='*', default=get_files('data/pkls'))

	args = a.parse_args()

	for input in args.inputs:
		frb = get_frb_name(input)
		output = f'output/{frb}_out.json'

		frb_data = get_data(input)
		calculate_data(frb_data, output)
