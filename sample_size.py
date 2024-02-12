'''
Creates a CSV file that describes the available data in data/table.csv.
Stores in data/sample.csv.
'''

import pandas as pd
import numpy as np

table_file = 'data/table.csv'
sample_file = 'data/sample.csv'

table = pd.read_csv(table_file)

yes = 'Y'
no = 'N'

have = lambda x: yes if not np.isnan(x) else no

rows = []
for index, row in table.iterrows():
	rows.append({
		'FRB': row['FRB'],
		'DM': have(row['DM_obs (pc cm^-3)']),
		'RM': have(row['RM_obs (rad/m^2)']),
		'Scattering Time Scale': have(row['Tau_obs (ms)']),
		'Host Properties': have(row['Host magnitude (AB)']),
	})

sample = pd.DataFrame.from_records(rows)

sample.to_csv(sample_file, index=False)

count = lambda lst, col: len([d for d in lst if d[col] == yes])

with open(sample_file, 'a') as f:
	f.write(f'Total,{count(rows, "DM")},{count(rows, "RM")},{count(rows,"Scattering Time Scale")},{count(rows, "Host Properties")}')