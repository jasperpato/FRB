'''
Visualises the collected data.
'''

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import json


def plot_residuals(xs, ys, params, rms, ax):
	'''
	Plot residuals / RMS as a scatter plot.
	'''
	residuals = ys - np.array([sum_exgauss(x, *params) for x in xs])
	ax.scatter(xs, residuals / rms, color='black', alpha=0.3)
	ax.set_ylabel('Residuals / RMS')


def plot(xs, ys, params, rms, label=''):
	'''
	Plot the sum of exgausses with given params. Also plot individual exgauss components and original FRB.
	'''
	fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2]})
	fig.subplots_adjust(hspace=0)

	plot_residuals(xs, ys, params, rms, ax[0])
	
	# FRB
	ax[1].plot(xs, ys, label='FRB', color='red')
	
	# fitted curve
	ax[1].plot(xs, [sum_exgauss(x, *params) for x in xs], label=label, color='black')
	
	# components
	for i in range(0, len(params)-1, 3):
		ax[1].plot(xs, [exgauss(x, *params[i:i+3], params[-1]) for x in xs], linestyle='dotted', color='blue')
	
	ax[1].set_ylabel('Intensity')
	plt.xlabel('Time (us)')
	fig.legend()


def plot_fitted(xs, ys, rms, data_file, ns=None):
	'''
	Plots the fitted curved found in the json data. Can be filtered with an optional parameter ns: a list of ns to plot.
	'''
	with open(data_file, 'r') as f:
		data = json.load(f)

	for n in data:
		if not ns or int(n) in ns:	
			params = data[n]['params']
			plot(xs, ys, params, rms, f'fit N={n}')


if __name__ == '__main__':
	import sys

	frb = 'data/single_FRB_221106_I_ts_343.0_64_avg_1_200.npy'
	
	ys = np.load(frb)
	rms = np.std(ys[:3700]) # manually get baseline rms
	
	ys = ys[3700:4300] # manually extract burst
	xs = range(len(ys))

	data_file = get_data_file(sys.argv)
	if not data_file: exit()

	plot_fitted(xs, ys, rms, data_file, get_nums(sys.argv))

	plt.show(block=True)