'''
Visualises the curve fitted FRBs.
'''

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import json
import os
import globalpars
from PIL import Image
import math

def plot_single_fit(xs, ys, params):
	'''
	Plot fit model on a separate figure. Used for displaying the initial params
	against the raw FRB.
	'''
	fig, ax = plt.subplots(1, 1)
	ax.plot(xs, ys, color='red', label='Smoothed FRB')
	ax.plot(xs, [exgauss(x, *params) for x in xs], color='black', label='Estimated params')
	fig.suptitle('Estimated Parameters')
	fig.legend()


def plot_residuals(xs, ys, params, rms, ax):
	'''
	Plot residuals / RMS as a scatter plot.
	'''
	residuals = ys - np.array([exgauss(x, *params) for x in xs])
	ax.scatter(xs, residuals / rms, color='black', alpha=0.3)
	ax.set_ylabel('Residuals / RMS')


def _plot(xs, ys, data, rms, n, low_i, frbname):
	'''
	Plot the sum of exgausses with given params. Also plot individual exgauss components and original FRB.
	'''
	fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2]})
	fig.subplots_adjust(hspace=0)

	ax[1].set_ylabel('Intensity')
	ax[1].set_xlabel('Time (ms)')

	params = data['Params']
	low, high = data['Burst range']

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

	x2.set_xticks(xs[[max(0, low - low_i), min(len(xs)-1, high - low_i)]])
	x2.set_xticklabels([])
	
	fig.suptitle(f'{frbname} R^2={data["Adjusted R^2"]:.2f}')
	fig.legend()


def plot_fitted(xs, ys, rms, data, n, frbname, show_initial=False):
	'''
	Plots the fitted curved found in the json data. Can be filtered with an optional parameter ns: a list of ns to plot.
	'''
	d = data['data'][n]

	# trunctate tails
	low, high = d['Burst range']
	width = high - low
	low, high = max(0, low - width * globalpars.N_WIDTHS), min(len(xs), high + width * globalpars.N_WIDTHS)

	xs, ys = xs[low:high], ys[low:high]

	if show_initial:
		plot_single_fit(xs, ys, d['Initial params'])

	_plot(xs, ys, d, rms, n, low, frbname)


def combine(dir, ncols=4, max_imgs=8, name='combined'):
	entries = [entry for entry in get_files(dir) if name not in entry][:max_imgs]
	n = len(entries)
	rows, cols = math.ceil(n / ncols), ncols
	fig, axs = plt.subplots(rows, cols)

	for i, entry in enumerate(entries):
		img = np.asarray(Image.open(entry))
		r, c = i // ncols, i % ncols

		ax = axs[r][c]
		ax.imshow(img)
		ax.set_axis_off()

	for i in range(n, rows * cols):
		r, c = i // ncols, i % ncols
		axs[r][c].set_visible(False)

	fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
	fig.savefig(f'{dir}/{name}')


def plot_r2(frbname, data):
	'''
	Plot N vs Adjusted R^2 for a given frb.
	'''
	data = [(int(n), d['Adjusted R^2']) for n, d in data.items()]
	ns, r2s = [n for n, _ in data], [r2 for _, r2 in data]

	plt.figure()
	plt.plot(ns, r2s)
	plt.title(f'{frb}')
	plt.xlabel('N')
	plt.xticks(range(max(ns) + 1))
	plt.ylabel('Adjusted R^2')
	plt.savefig(f'figs/adjusted_r2/{frbname}_r2')
	plt.close()


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--show-initial', action='store_true')
	a.add_argument('--show', action='store_true')
	a.add_argument('--combine-only', action='store_true')
	a.add_argument('--threshold', action='store_true')
	a.add_argument('--r2', action='store_true')
	a.add_argument('inputs', nargs='*', default=get_files('data/pkls'))

	args = a.parse_args()

	if args.combine_only:
		combine('figs/fits')
		exit()

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
			n = data['Threshold R^2'] if args.threshold else data['Max R^2']

			plot_fitted(frb_data.tmsarr, frb_data.it, frb_data.irms, data, n, frb, args.show_initial)
				
			if not (args.show or args.show_initial):
				plt.savefig(f'figs/fits/{frb}')
				plt.close()
	
	if args.show or args.show_initial:
		plt.show(block=True)