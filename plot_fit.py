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
	residuals = ys - np.array([exgauss(x, *params) for x in xs])
	ax.scatter(xs, residuals / rms, color='black', alpha=0.3)
	ax.set_ylabel('Residuals / RMS')


def plot(xs, ys, data, rms, label=''):
	'''
	Plot the sum of exgausses with given params. Also plot individual exgauss components and original FRB.
	'''
	fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2]})
	fig.subplots_adjust(hspace=0)

	params = data['params']
	low, high = data['burst_range']

	plot_residuals(xs, ys, params, rms, ax[0])
	
	# FRB
	ax[1].plot(xs, ys, label='FRB', color='red')
	
	# fitted curve
	ax[1].plot(xs, [exgauss(x, *params) for x in xs], label=label, color='black')
	
	# components
	for i in range(0, len(params)-1, 3):
		ax[1].plot(xs, [exgauss(x, *params[i:i+3], params[-1]) for x in xs], linestyle='dotted', color='blue')

	# burst width
	ax[1].axvline(low, color='green')
	ax[1].axvline(high, color='green')
	
	ax[1].set_ylabel('Intensity')
	plt.xlabel('Time (us)')
	fig.legend()


def plot_fitted(xs, ys, rms, data_file, ns=None):
	'''
	Plots the fitted curved found in the json data. Can be filtered with an optional parameter ns: a list of ns to plot.
	'''
	with open(data_file, 'r') as f:
		data = json.load(f)['data']

	for n in data:
		if not ns or int(n) in ns:	
			d = data[n]
			plot(xs, ys, d, rms, f'fit N={n}')


if __name__ == '__main__':
	import sys

	# frb = 'data/single_FRB_221106_I_ts_343.0_64_avg_1_200.npy'
	frb = 'data/221106.pkl'
	
	data = 'data/data.json'
	
	for x in sys.argv:
		if '.json' in x:
			data = x
	
	# range = 
	# ys = np.load(frb)
	# xs = range(len(ys))

	name, xs, ys, timestep, rms = get_data(frb)
	# xs = range(len(ys))

	rms = np.std(ys[:3700]) # manually get baseline rms

	# ys = ys[3700:4300] # manually extract burst for plotting
	# xs = xs[3700:4300]
	
	ys = ys[7600:8500]
	xs = range(len(ys))

	plot_fitted(xs, ys, rms, data, get_nums(sys.argv))

	plt.show(block=True)