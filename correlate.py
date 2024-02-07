'''
By default takes a single column name as an argument and plots the correlation between that
column and all other columns that do not match an IGNORE pattern. Saves the correlation plots
in /figs/correlations/col_name.

--plot-hists: shows the histograms of randomly shuffling one column and finding
spearman and pearson coefficients.
'''


from scipy.stats import spearmanr, pearsonr
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
import os


def remove_nans(a, b, *args):
	'''
	Returns copies of the input lists, with elements removed from both lists in the positions
	where there is a nan in one of the lists.
	Also removes corresponding elements from remaining args.
	'''
	a, b = np.array(a), np.array(b)
	is_nan = np.logical_or(np.isnan(a), np.isnan(b))
	return a[~is_nan], b[~is_nan], *[(c[~is_nan] if c is not None else None) for c in args]


def collapse(str, c):
	'''
	Collapse multiple consecutive instances of c in str with a single instance.
	'''
	res = ''
	for ch in str:
		if ch != c or res[-1] != c:
			res += ch
	return res


def file_name(name):
	'''
	Replace space characters and other brackets with underscores.
	'''
	for c in '()/ ':
		name = name.replace(c, '_')
	return collapse(name.strip('_'), '_')


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


def correlate(x0, x1, name0, name1, x0_err=None, x1_err=None, plot_hists=False, save_fig=False, n=1000):
	'''
	Plot correlation coefficients.
	'''
	x0, x1, x0_err, x1_err = remove_nans(x0, x1, x0_err, x1_err)
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
		plt.rc('axes', labelsize=16, titlesize=16)
		fig, ax0 = plt.subplots()
	
	# plot scatter of x0 vs x1
	if x0_err is not None:
		ax0.errorbar(x0, x1, xerr=x0_err, yerr=x1_err, fmt='o', ecolor='lightgrey')
	else:
		ax0.scatter(x0, x1)
	ax0.set_xlabel(name0)
	ax0.set_ylabel(name1)
	ax0.set_title(f'    Spearman: {spear:.2f} [{s_err:.2f}]\n    Pearson:     {pear:.2f} [{p_err:.2f}]    (sample={len(x0)})', loc='left') # horizontalalignment='left')
	
	if save_fig:
		name0, name1 = file_name(name0), file_name(name1)

		# make directory if not made
		try: os.mkdir(f'figs/correlations/{name0}')
		except: pass

		fig.savefig(f'figs/correlations/{name0}/{name0}_{name1}', bbox_inches='tight')
		plt.close(fig)


def get_error_col(data, col_name):
	'''
	Return error column corresponding to col_name if present in data, else array of zeroes.
	Assumes error column is the neighbouring column in data and contains 'error'.
	'''
	zero_err = np.zeros(len(data))
	next_i = data.columns.get_loc(col_name) + 1
	
	if next_i == len(data.columns): return zero_err
	
	next_col = data.iloc[:, next_i]
	return next_col if 'error' in next_col.name else zero_err


if __name__ == '__main__':
	from argparse import ArgumentParser

	DEFAULTS = [
		'log(DM_MW) (NE2001)', 'log(DM_obs)', 'log(DM_IGM)', 'log(DM_ex)',
		'log(abs(RM_obs))', 'log(abs(RM_MW))', 'log(abs(RM_ex))',
		'log(Tau_MW)', 'log(Tau_obs)', 'log(Tau_ex)'
	]

	DEFAULTS_NO_LOG = [
		'DM_MW (NE2001)', 'DM_obs (pc cm^-3)', 'DM_IGM', 'DM_ex (NE2001)',
		'abs(RM_obs)', 'abs(RM_MW)', 'abs(RM_ex)',
		'Tau_MW (ms)', 'Tau_obs (ms)', 'Tau_ex (ms)'
	]

	IGNORE = 'Index|FRB|RA|DEC|error|Repeater|AGN'

	a = ArgumentParser()
	a.add_argument('targets', nargs='*', default=DEFAULTS)
	a.add_argument('--table', default='data/table.csv')
	a.add_argument('--no-log', action='store_true')
	a.add_argument('--plot-hists', action='store_true')
	a.add_argument('--single', action='store_true')
	a.add_argument('--show', action='store_true')

	args = a.parse_args()

	if args.no_log:
		args.targets = DEFAULTS_NO_LOG

	# data = pd.read_csv('data/table.csv')
	data = pd.read_csv(args.table)

	cols = [col for col in data.columns if not re.search(IGNORE, col)]
	errs = [get_error_col(data, col) for col in cols]

	# single correlation of two quantities
	if args.single and len(args.targets) == 2:
		t0, t1 = args.targets
		print(f'Correlating {t0} and {t1}')
		err0, err1 = get_error_col(data, t0), get_error_col(data, t1)
		correlate(data[t0], data[t1], t0, t1, err0, err1, plot_hists=args.plot_hists, save_fig=not args.show)

	elif len(args.targets) >= 1:
		for t in args.targets:
			err = get_error_col(data, t)

			for i in range(len(cols)):
				col1, err1 = cols[i], errs[i]
				if col1 != t:
					if col1 == 'abs(Galactic lat)' or ('log' in t and 'log' in col1) or ('log' not in t and 'log' not in col1):
						print(f'Correlating {t} and {col1}')
						correlate(data[t], data[col1], t, col1, err, err1, plot_hists=args.plot_hists, save_fig=not args.show)

	if args.show:
		plt.show(block=True)