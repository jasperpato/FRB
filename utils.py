from scipy.stats import exponnorm
import numpy as np

def exgauss(x, a, u, sd, ts):
	'''
	Returns the single value f(x) of the exponnorm function with params (a, u, sd, ts)

	a: vertical scale param
	u: mean of the gaussian
	sd: standard deviation of the gaussian
	ts: scattering timescale, equivalent to 1/l where l is the mean of the exponential
	'''
	K = ts / sd
	return a * exponnorm.pdf(x, K, u, sd)


def sum_exgauss(x, *args):
	'''
	Returns the single value f(x) of the sum of n exponnorm functions with individual params listed in args, followed by a single param ts
	eg. sum_exgauss(x, a0, u0, sd0, a1, u1, sd1, ts)
	'''
	return sum(exgauss(x, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3))


def rsquared(xs, ys, params):
	'''
	Returns R squared value for a set of parameters.
	'''
	residuals = ys - np.array([sum_exgauss(x, *params) for x in xs])
	ss_res = np.sum(residuals ** 2)
	ss_tot = np.sum((ys - np.mean(ys)) ** 2)
	return 1 - (ss_res / ss_tot)


def adjusted_rsquared(xs, ys, params):
	'''
	Returns adjust R squared value for a set of parameters.
	'''
	rs = rsquared(xs, ys, params)
	return 1 - (1 - rs) * (len(xs) - 1) / (len(xs) - len(params) - 1)


def get_nums(lst):
	'''
	Extract integers from command line arguments.
	'''
	nums = []
	for x in lst:
		try: nums.append(int(x))
		except: pass
	return nums


def get_data_file(lst):
	'''
	Extract data_file from command line arguments.
	'''
	for x in lst:
		if '.json' in x:
			return x