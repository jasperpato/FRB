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


def model_burst_range(xs, params, area=globalpars.MODEL_CENTRE_AREA):
	'''
	Get the burst range of a fitted curve by iterating through xs and finding the
	central range containing `width` proportion of the total area under the curve.
	'''
	total_area = integral(xs[-1], *params)
	low_area = (1 - area) / 2 * total_area
	high_area = (1 + area) / 2 * total_area

	i = 0
	while integral(xs[i], *params) < low_area: i += 1
	low = i
	i = len(xs) - 1
	while integral(xs[i], *params) > high_area: i -= 1

	return low, i


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


def replace_nan(arr):
	'''
	Replace NaN with mean of neighbours. 'Pads' the array with leading and trailing
	zeroes.
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
	Returns linear polarisation fluence and associated error.
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


def calculate_data(frb_data, data_file):
	'''
	Calculate the relevant data for a curve fitted FRB.
	'''
	with open(data_file, 'r') as f:
		data = json.load(f)

	low, high = data['range']

	xs = frb_data.tmsarr
	it = frb_data.it
	timestep = frb_data.tresms


	for n, d in data['data'].items():
		d['adjusted_R^2'] = adjusted_rsquared(xs[low:high], it[low:high], d['params'])
		
		d['burst_range'] = (b := model_burst_range(xs, d['params']))
		d['burst_width'] = (b[1] - b[0]) * timestep

		# use model bounds
		it_bounded = it[b[0]:b[1]]
		lt = replace_nan(frb_data.lt[b[0]:b[1]])
		pt = replace_nan(frb_data.pt[b[0]:b[1]])
		elts = replace_nan(frb_data.elts[b[0]:b[1]])
		epts = replace_nan(frb_data.epts[b[0]:b[1]])
		
		d['fluence'] = (f := [np.sum(it_bounded), frb_data.irms * len(xs) ** 0.5]) # fluence and associated error
		d['linear polarisation fluence'] = (l := polarisation_fluence(lt, elts))
		d['total polarisation fluence'] = (p := polarisation_fluence(pt, epts))
		
		d['linear polarisation fraction'] = polarisation_fraction(*l, *f)
		d['total polarisation fraction'] = polarisation_fraction(*p, *f)
		
		d['timescale'] = d['params'][-1]
		
		data['data'][n] = dict(sorted(d.items()))

	data['optimum'] = max(data['data'].keys(), key=lambda n: data['data'][n]['adjusted_R^2'])
	
	with open(data_file, 'w') as f:
		json.dump(data, f)


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--input', default='data/221106.pkl')
	a.add_argument('--output', default=None)

	args = a.parse_args()

	if not args.output:
		frb = get_frb(args.input)
		args.output = f'output/{frb}_out.json'

	frb_data = get_data(args.input)
	calculate_data(frb_data, args.output)

	# print(replace_nan(np.array([np.nan,1,np.nan,2,3,np.nan,np.nan,5,np.nan,np.nan])))