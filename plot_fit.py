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
	ax[1].axvline(xs[low], color='green')
	ax[1].axvline(xs[high], color='green')
	
	ax[1].set_ylabel('Intensity')
	plt.xlabel('Time (us)')
	fig.legend()


def plot_fitted(xs, ys, rms, data, ns=None):
	'''
	Plots the fitted curved found in the json data. Can be filtered with an optional parameter ns: a list of ns to plot.
	'''
	for n in data:
		if not ns or int(n) in ns:	
			d = data[n]
			plot_single_fit(xs, ys, d['initial_params'])
			plot(xs, ys, d, rms, f'fit N={n}')


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--input', default='data/221106.pkl')
	a.add_argument('--output', default=None)
	a.add_argument('--print', action='store_true')
	a.add_argument('N', type=int, nargs='*')

	args = a.parse_args()
	if not args.output:
		args.output = default_output(args.input)

	name, xs, ys, timestep, rms = get_data(args.input)

	low, high = raw_burst_range(ys)
	
	ys = ys[low:high]
	xs = xs[low:high]

	with open(args.output, 'r') as f:
		data = json.load(f)

	if args.print:
		print_summary(data, timestep)

	if not args.N:
		args.N.append(int(data['optimum']))

	plot_fitted(xs, ys, rms, data['data'], args.N)
	plt.show(block=True)