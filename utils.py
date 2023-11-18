from scipy.stats import exponnorm
import numpy as np


def exgauss(x, *args):
	'''
	Returns the single value f(x) of the sum of n exGaussian functions with individual
	params listed in args, followed by a single param ts.
	eg. exgauss(x, a0, u0, sd0, a1, u1, sd1, ts)
	'''
	exgauss = lambda x, a, u, sd, ts: a * exponnorm.pdf(x, ts / sd, loc=u, scale=sd)
	return sum(exgauss(x, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3))


def integral(x, *args):
	'''
	Integral from -inf to x of the sum of exGaussians.
	'''
	single_integral = lambda x, a, u, sd, ts: a * exponnorm.cdf(x, ts / sd, loc=u, scale=sd)
	return sum(single_integral(x, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3))


def burst_range(params, len_xs, width=0.99):
	'''
	Get the burst range of a fitted curve by iterating through xs and finding the
	central range containing `width` proportion of the total area under the curve.
	'''
	xs = np.arange(len_xs)

	total_area = integral(xs[-1], *params)
	low_area = (1 - width) / 2 * total_area
	high_area = (1 + width) / 2 * total_area

	x = 0
	while integral(x, *params) < low_area: x += 1
	low = x
	while integral(x, *params) < high_area: x += 1

	return low, x


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


def get_nums(lst):
	'''
	Extract integers from command line arguments.
	'''
	nums = []
	for x in lst:
		try: nums.append(int(x))
		except: pass
	return nums
