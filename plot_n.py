import json
import matplotlib.pyplot as plt


def plot_data(data_file):
	'''
	plot ns and such
	'''
	with open(data_file, 'r') as f:
		data = json.load(f)
	fig, ax = plt.subplots(1, 1)
	ax.plot(data.keys(), [d['rsquared'] for d in data.values()])


if __name__ == '__main__':
	data_file = 'data/second_pass.json'
	
	plot_data(data_file)
	plt.show(block=True)