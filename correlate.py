'''
Correlates all columns from table except any matching an IGNORE pattern.
Prints correlations in order of abs of spearman coefficient.
'''


from scipy.stats import spearmanr, pearsonr
import pandas as pd
import re
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt


IGNORE = ('Index', 'FRB', 'RA', 'DEC', 'error', 'Repeater', 'AGN')
IGNORE = '|'.join(IGNORE)

# PAIRS = [
# 	('Galactic lat','DM_MW (NE2001)'),
# 	('Galactic lat','DM_IGM'),
# 	('Galactic lat','DM_obs (pc cm^-3)'),
# ]


def remove_nans(a, b):
	'''
	Returns copies of the input lists, with elements removed from both lists in the positions
	where there is a nan in one of the lists.
	'''
	a, b = np.array(a), np.array(b)
	is_nan = np.logical_or(np.isnan(a), np.isnan(b))
	return a[~is_nan], b[~is_nan]


def remove_negs(a, b):
	'''
	Returns copies of the input lists, with elements removed from both lists in the positions
	where there is a negative value in the first list.
	'''
	a, b = np.array(a), np.array(b)
	is_neg = a < 0
	return a[~is_neg], b[~is_neg]


def file_name(name):
	'''
	Remove units in brackets and replace space characters and other brackets with underscores.
	'''
	if ' (' in name: name = name.split(' (')[0]
	for c in '()/ ':
		name = name.replace(c, '_')
	return name


def correlate(x0, x1, name0, name1, neg_policy='', n=1000, bins=20):
	'''
	Plot correlation coefficients.
	'''
	x0, x1 = remove_nans(x0, x1)
	if neg_policy == 'remove': x0, x1 = remove_negs(x0, x1)
	if len(x0) < 3: return

	get_c = lambda x2, method='spearman': spearmanr(x0, x2, nan_policy='omit').statistic if method == 'spearman' else pearsonr(x0, x2).statistic
	
	xs, ss, ps = [], [], []
	for _ in range(n):
		x2 = np.array(x1.copy())
		np.random.shuffle(x2)
		
		xs.append(x2)
		ss.append(get_c(x2))
		ps.append(get_c(x2, 'pearson'))

	spear = get_c(x1)
	pear = get_c(x1, 'pearson')

	plt.rc('font', size=8)
	fig, axs = plt.subplots(2, 2, figsize=(8, 8))
	ax0, ax1, ax2, ax3 = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
	ax3.axis('off')

	get_pos = lambda arr, x=0.8: min(arr) + x * (max(arr) - min(arr))
	
	# plot scatter of x0 vs x1
	ax0.scatter(x0, x1)
	ax0.set_xlabel(name0)
	ax0.set_ylabel(name1)
	ax0.text(get_pos(x0, 0.6), get_pos(x1), f'Spearman: {spear:.4f}\nPearson:  {pear:.4f}')
	ax0.set_title(f'{name0} vs {name1}')

	def plot_hist(ax, arr, name):
		'''
		Plot coefficients as a histogram, display mean and std.
		'''
		n, _, _ = ax.hist(arr, bins, density=True)
		ax.text(get_pos(arr, 0.6), get_pos(n), f'mean: {np.mean(arr):.4f}\nstd: {np.std(arr):.4f}')
		ax.set_title(f'{name} coefficient histogram')
		ax.set_xlabel('Coefficient')
		ax.set_ylabel('Frequency')

	plot_hist(ax1, ss, 'Spearman')
	plot_hist(ax2, ps, 'Pearson')

	name0, name1 = file_name(name0), file_name(name1)
	fig.savefig(f'figs/correlations/{name0}/{name0}_{name1}')


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('target')
	a.add_argument('--neg-policy')

	args = a.parse_args()

	data = pd.read_csv('data/table.csv')
	cols = [col for col in data.columns if not re.search(IGNORE, col)]

	corrs = []
	for col1 in cols: # combinations(cols, 2):
		if col1 != args.target:
			correlate(data[args.target], data[col1], args.target, col1, neg_policy=args.neg_policy)

	# plt.show(block=True)
			