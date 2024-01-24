from scipy.stats import exponnorm
import numpy as np
import pickle
import os


def exgauss(xs, *args):
	'''
	Returns the single value f(x) of the sum of n exGaussian functions with individual
	params listed in args, followed by a single param ts.
	eg. exgauss(x, a0, u0, sd0, a1, u1, sd1, ts)
	'''
	exgauss = lambda xs, a, u, sd, ts: a * exponnorm.pdf(xs, ts / sd, loc=u, scale=sd)
	return np.sum([exgauss(xs, *args[i:i+3], args[-1]) for i in range(0, len(args)-1, 3)], axis=0)


def get_data(pkl_file):
	'''
	Return named tuple containing FRB data.
	'''
	with open(pkl_file, 'rb') as f:
		return pickle.load(f)
	

def get_files(dir):
	'''
	Get all input files from a directory.
	'''
	return [f'{dir}/{f}' for f in os.listdir(dir)]
	