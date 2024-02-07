import pandas as pd
import numpy as np

# data = pd.read_csv('data/htr_sample.csv')

# data['log(abs(RM))'] = np.log10(np.abs(data['RM']))
# data['log(abs(RM_MW))'] = np.log10(np.abs(data['RM_MW']))
# data['log(abs(RM_EG))'] = np.log10(np.abs(data['RM_EG']))

# data['log(abs(RM)) error'] = np.abs(data['RM error'] / data['RM'] / np.log(10))
# data['log(abs(RM_MW)) error'] = np.abs(data['RM_MW error'] / data['RM_MW'] / np.log(10))
# data['log(abs(RM_EG)) error'] = np.abs(data['RM_EG error'] / data['RM_EG'] / np.log(10))

# data.to_csv('data/htr_sample.csv')

data0 = pd.read_csv('data/htr_sample.csv', index_col=0)
data1 = pd.read_csv('data/table.csv', index_col=0)

for frb in data0['FRB']:
	print(f'''
{frb}
{data0.loc[data0['FRB'] == frb]['log(1+z)']}
{data1.loc[data1['FRB'] == frb]['log(1+z)']}
{data0.loc[data0['FRB'] == frb]['log(abs(RM))']}
{data1.loc[data1['FRB'] == frb]['log(abs(RM_obs))']}'''
	)