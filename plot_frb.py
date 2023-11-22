import matplotlib.pyplot as plt
from utils import *
from argparse import ArgumentParser
from os.path import basename

a = ArgumentParser()
a.add_argument('--input', default='frb_data/221106.pkl')
a.add_argument('--save', action='store_true')
args = a.parse_args()

name, xs, ys, timestep, rms = get_data(args.input)
low, high = raw_burst_range(ys)

ys = ys[low:high]
xs = xs[low:high]

plt.plot(xs, ys, color='red')

if args.save:
	plt.savefig(f'figs/frbs/{basename(args.input)[:-4]}.png')

plt.show(block=True)