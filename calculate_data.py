from utils import *
import json
import numpy as np
from scipy.stats import exponnorm
import globalpars


def integral(x, *args):
	'''
	Integral from -inf to x of the sum of exGaussians.
	'''
	single_integral = lambda x, a, u, sd, ts: a * exponnorm.cdf(x, ts / sd, loc=u, scale=sd)
	return sum(single_integral(x, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3))


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
	while integral(xs[i], *params) < high_area: i += 1

	return low, i


def rsquared(xs, ys, params):
	'''
	Returns R squared value for a set of parameters.
	'''
	residuals = ys - np.array([exgauss(x, *params) for x in xs])
	ss_res = np.sum(residuals ** 2)
	ss_tot = np.sum((ys - np.mean(ys)) ** 2)
	return 1 - (ss_res / ss_tot)


def adjusted_rsquared(xs, ys, params):
	'''
	Returns adjust R squared value for a set of parameters.
	'''
	rs = rsquared(xs, ys, params)
	return 1 - (1 - rs) * (len(xs) - 1) / (len(xs) - len(params) - 1)


def linear_pol_fraction(lt, elts, fluence, fluence_error):
	pass


def calculate_data(frb_data, data_file):
	'''
	Calculate the relevant data for a curve fitted FRB.
	'''
	xs = frb_data.tmsarr
	ys = frb_data.it
	timestep = frb_data.tresms

	with open(data_file, 'r') as f:
		data = json.load(f)

	for n, d in data['data'].items():
		d['adjusted_R^2'] = adjusted_rsquared(xs, ys, d['params'])
		d['burst_range'] = (b := model_burst_range(xs, d['params']))
		d['burst_width'] = (b[1] - b[0]) * timestep
		d['fluence'] = [sum(ys[b[0]:b[1]]), frb_data.irms * len(xs) ** 0.5] # fluence and associated error
		d['linear polarisation fraction'] = linear_pol_fraction(frb_data.lt, frb_data.elts, *d['fluence'])
		d['total polarisation fraction'] = 0
		d['timescale'] = d['params'][-1]
		data['data'][n] = dict(sorted(d.items()))

	with open(data_file, 'w') as f:
		json.dump(data, f)


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--input', default='frb_data/221106.pkl')
	a.add_argument('--output', default=None)

	args = a.parse_args()

	if not args.output:
		frb = get_frb(args.input)
		args.output = f'output/{frb}_out.json'

	frb_data = get_data(args.input)
	calculate_data(frb_data, args.output)

	