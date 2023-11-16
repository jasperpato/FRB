import json
import matplotlib.pyplot as plt
from utils import *


def plot_data(data_file):
	'''
	plot n against adjusted R squared value.
	'''
	with open(data_file, 'r') as f:
		data = json.load(f)

	plt.plot(data.keys(), [d.get('adjusted', 0) for d in data.values()])
	plt.xlabel('N')
	plt.ylabel('Adjusted R^2')
	plt.title('Adjusted R^2 against Number of exGaussians')

if __name__ == '__main__':
	import sys
	data_file = get_data_file(sys.argv)
	if not data_file: exit()
	plot_data(data_file)
	plt.show(block=True)