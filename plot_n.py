import json
import matplotlib.pyplot as plt
from utils import *


def plot_data(data_file, plot_timescale=False):
	'''
	Plot adjusted R squared and scattering timescale against n.
	'''
	with open(data_file, 'r') as f:
		data = json.load(f)['data']

	# plot adjusted R^2 against n

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
	import sys
	
	data_files = [x for x in sys.argv if '.json' in x]
	if not data_files:
		exit()

	for data_file in data_files:
		plot_data(data_file)

	plt.show(block=True)