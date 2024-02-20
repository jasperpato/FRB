import pandas as pd
import numpy as np

DPS = 2

data_file = 'data/table.csv'
paper_file = 'data/paper_data.csv'

data = pd.read_csv(data_file)

drop = np.logical_or(data['Components'] == 0, data['Tau_obs (ms)'] == 0)
data = data.drop(data[drop].index)

def combine(val, err):
	if np.isnan(err).all(): return f'{val:.{DPS}f}'
	return f'{val:.{DPS}f} \(\pm\) {err:.{DPS}f}'

ufunc = np.vectorize(combine)

data['DM_obs (pc cm^-3)'] = ufunc(data['DM_obs (pc cm^-3)'], data['DM_obs error'])
data['DM_MW (NE2001)'] = ufunc(data['DM_MW (NE2001)'], data['DM_MW error (NE2001)'])
data['Tau_obs (ms)'] = ufunc(data['Tau_obs (ms)'], data['Tau_obs error'])

data['Galactic lat'] = np.round(data['Galactic lat'], DPS)

data = data[[
	'FRB',
	'DM_obs (pc cm^-3)',
	'DM_MW (NE2001)',
	'Galactic lat',
	'Tau_obs (ms)',
	'Components'
]]

data.to_csv(paper_file, index=False)
