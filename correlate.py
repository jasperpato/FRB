'''
Correlates all columns from table except any matching an IGNORE pattern.
Prints correlations in order of abs of spearman coefficient.
'''


from scipy.stats import spearmanr, pearsonr
import pandas as pd
import re
import numpy as np
from itertools import combinations


IGNORE = ('Index', 'FRB', 'RA', 'DEC', 'error', 'Repeater', 'AGN')
IGNORE = '|'.join(IGNORE)


def remove_nans(a, b):
	'''
	Returns copies of the input lists, with elements removed from both lists in the positions
	where there is a nan in one of the lists.
	'''
	a, b = np.array(a), np.array(b)
	is_nan = np.logical_or(np.isnan(a), np.isnan(b))
	return a[~is_nan], b[~is_nan]


def correlate(x0, x1, method='spearman', n=1000):
	'''
	Find correlation coefficient along with mean and standard deviation.
	'''
	get_c = lambda x2: spearmanr(x0, x2, nan_policy='omit').statistic if method == 'spearman' else pearsonr(x0, x2).statistic
	
	cs = []
	for _ in range(n):
		x2 = np.array(x1.copy())
		np.random.shuffle(x2)
		cs.append(get_c(x2))

	return get_c(x1), np.mean(cs), np.std(cs)


if __name__ == '__main__':
	data = pd.read_csv('data/table_new.csv')
	cols = [col for col in data.columns if not re.search(IGNORE, col)]

	corrs = []
	for col0, col1 in combinations(cols, 2):
		x0, x1 = remove_nans(data[col0], data[col1])
		if len(x0) > 2:
			s = correlate(x0, x1)
			p = correlate(x0, x1, method='pearson')
			if not np.all(np.isnan([s[0], p[0]])):
				corrs.append([col0, col1, *s, *p])

	# sort by sum of abs of correlation coefficients
	corrs.sort(key=lambda l: abs(l[2]) + abs(l[5]), reverse=True)

	s = ''
	for c in corrs:
		s += f'{c[0]}, {c[1]}\nSpearman: {c[2]:.4f} mean: {c[3]:.4f} std: {c[4]:.4f}\nPearson:  {c[5]:.4f} mean: {c[6]:.4f} std: {c[7]:.4f}\n\n'

	with open('output/correlations.txt', 'w') as f:
		f.write(s)
			