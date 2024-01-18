'''
Correlates all columns from table except any matching an IGNORE pattern.
Prints correlations in order of abs of spearman coefficient.
'''


from scipy.stats import spearmanr, pearsonr
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
import os


IGNORE = 'Index|FRB|RA|DEC|error|Repeater|AGN'


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


def get_pos(arr, p='x'):
	return (m:=min(arr)) + (0.6 if p == 'x' else 0.8) * (max(arr) - m)


def plot_hist(ax, arr, name, bins=20):
		'''
		Plot coefficients as a histogram, display mean and std.
		'''
		n, _, _ = ax.hist(arr, bins, density=True)
		ax.text(get_pos(arr, 'x'), get_pos(n, 'y'), f'mean: {np.mean(arr):.2f}\nstd: {np.std(arr):.2f}')
		ax.set_title(f'{name} coefficient histogram')
		ax.set_xlabel('Coefficient')
		ax.set_ylabel('Frequency')


def correlate(x0, x1, name0, name1, plot_hists=False, save_fig=False, n=1000):
	'''
	Plot correlation coefficients.
	'''
	x0, x1 = remove_nans(x0, x1)
	if len(x0) < 3: return

	def get_c(x2, method='spearman'):
		if method == 'spearman': return spearmanr(x0, x2, nan_policy='omit').statistic
		return pearsonr(x0, x2).statistic
	
	xs, ss, ps = [], [], []
	for _ in range(n):
		x2 = np.array(x1.copy())
		np.random.shuffle(x2)
		
		xs.append(x2)
		ss.append(get_c(x2))
		ps.append(get_c(x2, 'pearson'))

	spear, s_err = get_c(x1), np.std(ss)
	pear, p_err = get_c(x1, 'pearson'), np.std(ps)

	if plot_hists:
		plt.rc('font', size=8)
		fig, axs = plt.subplots(2, 2, figsize=(8, 8))
		ax0, ax1, ax2, ax3 = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
		ax3.axis('off')

		plot_hist(ax1, ss, 'Spearman')
		plot_hist(ax2, ps, 'Pearson')

	else:
		fig, ax0 = plt.subplots()
	
	# plot scatter of x0 vs x1
	ax0.scatter(x0, x1)
	ax0.set_xlabel(name0)
	ax0.set_ylabel(name1)
	ax0.set_title(f'{name0} vs {name1}')
	ax0.text(
		get_pos(x0, 'x'),
		get_pos(x1, 'y'),
		f'Spearman: {spear:.2f} ± {s_err:.2f}\nPearson:     {pear:.2f} ± {p_err:.2f}'
	)
	
	if save_fig:
		name0, name1 = file_name(name0), file_name(name1)

		# make directory if not made
		try: os.mkdir(f'figs/correlations/{name0}')
		except: pass

		fig.savefig(f'figs/correlations/{name0}/{name0}_{name1}')


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('targets', nargs='*', default=['DM_ex (NE2001)'])
	a.add_argument('--save', action='store_true')
	a.add_argument('--plot-hists', action='store_true')

	args = a.parse_args()

	data = pd.read_csv('data/table.csv')
	cols = [col for col in data.columns if not re.search(IGNORE, col)]

	# from scipy.optimize import curve_fit

	# params = [1, -20, 1]
	# f = lambda xs, a, b, c: a * (1+xs)**b + c
	# plt.plot(data['z'], f(data['z'], *params))
	# plt.scatter(data['z'], data['DM_ex (NE2001)'])
	# plt.show(block=True)

	# popt, pcov = curve_fit(f, data['z'], data['DM_ex (NE2001)'])
	# print(popt)
	# exit()

	if len(args.targets) == 1:
		t = args.targets[0]
		for col1 in cols:
			if col1 != t:
				correlate(data[t], data[col1], t, col1, args.plot_hists, args.save)

	elif len(args.targets) == 2:
		t0, t1 = args.targets
		correlate(data[t0], data[t1], t0, t1, args.plot_hists, args.save)

	if not args.save:
		plt.show(block=True)


			