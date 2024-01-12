'''
Correlates all columns from table except any matching an IGNORE pattern.
Prints correlations in order of abs of spearman coefficient.
'''


from scipy.stats import spearmanr, pearsonr
import pandas as pd
import re
import numpy as np
from itertools import combinations


IGNORE = ('Index', 'FRB', 'RA', 'DEC', 'error')
IGNORE = '|'.join(IGNORE)


def remove_nans(a, b):
	'''
	Returns copies of the input lists, with elements removed from both lists in the positions
	where there is a nan in one of the lists.
	'''
	a, b = np.array(a), np.array(b)
	is_nan = np.logical_or(np.isnan(a), np.isnan(b))
	return a[~is_nan], b[~is_nan]


if __name__ == '__main__':
	data = pd.read_csv('table.csv')
	cols = [col for col in data.columns if not re.search(IGNORE, col)]

	corrs = []
	for col0, col1 in combinations(cols, 2):
		try:
			spear = spearmanr(data[col0], data[col1], nan_policy='omit').statistic
			pear = pearsonr(*remove_nans(data[col0], data[col1])).statistic
			if not np.all(np.isnan([spear, pear])):
				corrs.append([col0, col1, spear, pear])
		except: pass

	# sort by sum of abs of correlation coefficients
	corrs.sort(key=lambda l: abs(l[2]) + abs(l[3]), reverse=True)

	s = ''
	for c in corrs:
		s += f'{c[0]}, {c[1]}\nSpearman: {c[2]:.4f}\nPearson:  {c[3]:.4f}\n\n'

	with open('output/correlations.txt', 'w') as f:
		f.write(s)
			