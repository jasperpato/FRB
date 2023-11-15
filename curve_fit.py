import numpy as np
from scipy.optimize import curve_fit
from utils import *
import json


def estimate_params(n, xmin, xmax, prev_popt):
	'''
	Returns an array of 3*n + 1 params that are reasonable initial guesses to fit the data, and an array of bounds of the params.
	Spaces out n exgaussians along range [xmin, xmax].
	'''
	params = np.zeros(3*n+1)
	params[0::3] = 0.5 # a
	params[1::3] = np.linspace(xmin, xmax, n+2, endpoint=True)[1:-1] # gaussian means
	params[2::3] = 10 / n # sd
	params[-1] = 5 # ts

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
	bounds[1::3] = [xmin, xmax] # mean within range
	bounds[2::3] = [0, np.inf] # sd positive
	bounds[-1] = [0, 50] # ts positive

	return params, bounds.T


def rsquared(xs, ys, params):
	'''
	Returns R squared value for a set of parameters popt
	'''
	residuals = ys - np.array([sum_exgauss(x, *params) for x in xs])
	ss_res = np.sum(residuals ** 2)
	ss_tot = np.sum((ys - np.mean(ys)) ** 2)
	return 1 - (ss_res / ss_tot)


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
			p0, bounds = estimate_params(n, xs[0], xs[-1], prev_popt)
			try:
				popt, _ = curve_fit(sum_exgauss, xs, ys, p0, bounds=bounds)
				d['rsquared'] = rsquared(xs, ys, popt)
				
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
	Returns the exgauss params with the highest r^2 value in data.
	'''
	params = np.array(max(data.values(), key=lambda d: d['rsquared'])['params'])
	params[1:-1:3] -= shift # redo the shift for the next pass
	return params


def fit(ys, data0, data1):
	'''
	Fit curves of different n values 
	'''
	data = _fit(ys, 14, 15, data0) # manually set n range for first pass
	popt = get_popt(data, len(ys) / 2)
	print(popt)
	_fit(ys, 1, 10, data1, popt) # second pass


if __name__ == '__main__':
	frb = 'data/single_FRB_221106_I_ts_343.0_64_avg_1_200.npy'
	data0 = 'data/first_pass.json'
	data1 = 'data/second_pass.json'
	
	ys = np.load(frb)[3700:4300] # manually extract burst
	ys[:175] = 0 # manually zero the tails to improve the fit
	ys[450:] = 0

	fit(ys, data0, data1)
