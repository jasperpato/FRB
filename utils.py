from scipy.stats import exponnorm

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
