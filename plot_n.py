import json
import matplotlib.pyplot as plt
from utils import *


def plot_data(data_file, plot_timescale=False):
	'''
	Plot adjusted R squared and scattering timescale against n.
	'''
	with open(data_file, 'r') as f:
		data = json.load(f)['data']

	plt.plot(data.keys(), [d['adjusted_R^2'] for d in data.values()])
	plt.xlabel('N')
	plt.ylabel('Adjusted R^2')
	plt.title('Adjusted R^2 against Number of exGaussians')

	if plot_timescale:
		plt.figure()
		plt.plot(data.keys(), [d['params'][-1] for d in data.values()])
		plt.xlabel('N')
		plt.ylabel('Scattering timescale')
		plt.title('Scattering timescale against Number of exGaussians')


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--data', default='data/data.json')
	args = a.parse_args()

	plot_data(args.data)
	plt.show(block=True)