'''
Plots the curve fitted FRBs and residuals.

By default plots the fits of every FRB pickle file found in /data/pkls and saves
the pngs in /figs/fits.

--r2: Plots the N vs adjusted R^2 plots and saves in /figs/adjusted_r2.
--threshold: Plot the minimum N that exceeds the threshold adjusted R^2 instead of the N that maximises adjusted R^2.
'''


import numpy as np
import matplotlib.pyplot as plt
from utils import *
import json
import globalpars
from PIL import Image
import math


def plot_single_fit(xs, ys, params, frbname):
	'''
	Plot a fitted curve given by params on a separate figure. Used for displaying the initial params on the FRB signal.
	'''
	fig, ax = plt.subplots(1, 1)
	ax.plot(xs, ys, color='red', label='Smoothed FRB')
	ax.plot(xs, [exgauss(x, *params) for x in xs], color='black', label='Estimated params')
	fig.suptitle(frbname)
	fig.legend()


def plot_residuals(xs, ys, params, rms, ax):
	'''
	Plot residuals / RMS as a scatter plot on ax.
	'''
	residuals = ys - np.array([exgauss(x, *params) for x in xs])
	ax.scatter(xs, residuals / rms, color='black', alpha=0.3)
	ax.set_ylabel('Residuals / RMS')


def plot_fitted(xs, ys, rms, data, n, frbname, show_initial=False):
	'''
	Plots the fitted curved found in the json data. Can be filtered with an optional parameter ns: a list of ns to plot.
	'''
	# trunctate tails
	low, high = data['Burst range']
	width = high - low
	low, high = max(0, low - width * globalpars.N_WIDTHS), min(len(xs), high + width * globalpars.N_WIDTHS)

	xs, ys = xs[low:high], ys[low:high]

	if show_initial:
		plot_single_fit(xs, ys, data['Initial params'], frbname)

	fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2]})
	fig.subplots_adjust(hspace=0)

	ax[1].set_ylabel('Intensity')
	ax[1].set_xlabel('Time (ms)')

	params = data['Params']
	low1, high1 = data['Burst range']

	plot_residuals(xs, ys, params, rms, ax[0])
	
	# FRB
	ax[1].plot(xs, ys, label='FRB', color='red')
	
	# fitted curve
	ax[1].plot(xs, exgauss(xs, *params), label=f'N={n}', color='black')
	
	# components
	for i in range(0, len(params)-1, 3):
		ax[1].plot(xs, exgauss(xs, *params[i:i+3], params[-1]), linestyle='dotted', color='blue')

	# burst width ticks
	x2 = ax[1].twiny()
	x2.tick_params(direction='inout')
	x2.set_xlim([xs[0], xs[-1]])

	x2.set_xticks(xs[[max(0, low1 - low), min(len(xs)-1, high1 - low)]])
	x2.set_xticklabels([])
	
	fig.suptitle(f'{frbname} R^2={data["Adjusted R^2"]:.2f}')
	fig.legend()


def plot_r2(frbname, data):
	'''
	Plot N vs Adjusted R^2 for a given FRB.
	'''
	data = [(int(n), d['Adjusted R^2']) for n, d in data.items()]
	ns, r2s = [n for n, _ in data], [r2 for _, r2 in data]

	plt.figure()
	plt.plot(ns, r2s)
	plt.title(f'{frb}')
	plt.xlabel('N')
	plt.xticks(range(max(ns) + 1))
	plt.ylabel('Adjusted R^2')
	plt.savefig(f'figs/adjusted_r2/{frbname}_r2', bbox_inches='tight')
	plt.close()


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--show-initial', action='store_true')
	a.add_argument('--show', action='store_true')
	a.add_argument('--choose-r2', default='No increase R^2')
	a.add_argument('--r2', action='store_true')
	a.add_argument('inputs', nargs='*', default=get_files('data/pkls'))

	args = a.parse_args()

	for input in args.inputs:
		frb = get_frb_name(input)
		output = f'output/{frb}_out.json'

		print(frb)

		frb_data = get_data(input)

		with open(output, 'r') as f:
			data = json.load(f)

		if args.r2:
			plot_r2(frb, data['data'])

		else:
			n = data[args.choose_r2]
			plot_fitted(frb_data.tmsarr, frb_data.it, frb_data.irms, data['data'][n], n, frb, args.show_initial)
	
	if args.show or args.show_initial:
		plt.show(block=True)
