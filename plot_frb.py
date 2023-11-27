'''
Script to plot a single raw FRB signal.
'''

import matplotlib.pyplot as plt
from utils import *
from argparse import ArgumentParser
from os.path import basename
from confsmooth import confsmooth


a = ArgumentParser()
a.add_argument('--input', default='data/221106.pkl')
a.add_argument('--save', action='store_true')
args = a.parse_args()

# name, xs, ys, timestep, rms = get_data(args.input)
frb_data = get_data(args.input)

xs = frb_data.tmsarr
ys = frb_data.lt
# rms = frb_data.elts

low, high = raw_burst_range(frb_data.it)

ys = ys[low:high]
xs = xs[low:high]

# smooth = confsmooth(ys, rms)
plt.plot(xs, ys, color='red')
# plt.plot(xs, smooth, color='blue')

if args.save:
	plt.savefig(f'figs/frbs/{basename(args.input)[:-4]}.png')

plt.show(block=True)