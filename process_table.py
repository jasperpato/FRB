'''
Complete host properties and FRB burst properties in CSV file.
'''

from astropy.coordinates import SkyCoord, FK5, Galactic
from pyne2001 import get_galactic_dm
from zdm import parameters, pcosmic
import numpy as np
from pygedm import dist_to_dm
import pandas as pd
import globalpars
from utils import *
import json
from frb.rm import galactic_rm, load_hutschenreuter2020


def split_angle(angle):
	'''
	Split angle string into components.
	'''
	x0, x1, x2 = [float(x) for x in angle.split(':')]
	if angle[0] == '-': x1, x2 = -x1, -x2
	return x0, x1, x2


def to_decimal(ra, dec):
	'''
	Convert RA and DEC from hours, minutes, seconds into decimal degrees.
	'''
	ra_hours, ra_mins, ra_secs = split_angle(ra)
	dec_deg, dec_mins, dec_secs = split_angle(dec)
	
	return (
		15 * ra_hours + 15 * ra_mins / 60 + 15 * ra_secs / 3600,
		dec_deg + dec_mins / 60 + dec_secs / 3600
	)


def to_galactic(ra, dec):
	'''
	Convert from FK5 coords to Galactic coords.
	'''
	sc = SkyCoord(ra=ra, dec=dec, unit='deg', frame=FK5, equinox='J2000')
	sc = sc.transform_to(Galactic())
	return sc.l.deg, sc.b.deg


def get_error(lon, lat, method='NE2001', nu=1, width=2, size=30):
	'''
	Estimate error for DM and SM by taking size samples within +/- width.
	'''
	lons = np.random.uniform(low=lon - width, high=lon + width, size=size)
	lats = np.random.uniform(low=lat - width, high=lat + width, size=size)

	func = lambda lon, lat: get_galactic_dm(lon, lat) if method == 'NE2001' else (
		np.array([q.value for q in dist_to_dm(lon, lat, globalpars.MW_DIAMETER, nu=nu)])
	)
	ufunc = np.frompyfunc(func, 2, 1)

	return np.std(ufunc(lons, lats))


def get_igm_error(z, state, width=1e-2, size=30):
	'''
	Estimate error for DM IGM.
	'''
	zs = np.random.uniform(low=z - width, high=z + width, size=size)
	dms = [pcosmic.get_mean_DM(np.array([z]), state)[0] for z in zs]
	return np.std(dms)


def complete_row(row, state):
	'''
	Complete the missing columns of a row.
	'''
	ra, dec = 'RA', 'DEC'
	glon, glat = 'Galactic lon', 'Galactic lat'

	r, d = to_decimal(row[ra], row[dec])
	row[glon], row[glat] = to_galactic(r, d)

	if not np.isnan([row[glon], row[glat]]).any():
		# DM NE2001 and error
		row['DM_MW (NE2001)'] = get_galactic_dm(row[glon], row[glat])
		row['DM_MW error (NE2001)'] = get_error(row[glon], row[glat], method='NE2001')

		# RM_MW and error
		coord = SkyCoord(ra=r, dec=d, unit='deg', frame=FK5, equinox='J2000')
		rm, rm_error = galactic_rm(coord)
		row['RM_MW (rad/m^2)'], row['RM_MW error'] = rm.value, rm_error.value

	# DM IGM and error
	if not np.isnan(row['z']):
		row['DM_IGM'] = pcosmic.get_mean_DM(np.array([row['z']]), state)[0]
		row['DM_IGM error'] = get_igm_error(row['z'], state)

	return row


def complete_burst_properties(frb, data, output, frb_data):
	'''
	Search for FRB in data and fill in burst properties.
	'''
	i = data.index[data['FRB'] == frb].to_list()[0]
	freq = np.mean(frb_data.fmhzarr) * 1e-3 # convert to GHz

	# burst properties
	b = 'Burst width (ms)'
	data.at[i, b] = output[b]

	s = 'Scattering timescale (ms)'
	data.at[i, s], data.at[i, 'Scattering timescale error'] = output[s]

	l = 'Linear polarisation fraction'
	data.at[i, l], data.at[i, 'Linear polarisation fraction error'] = output[l]

	t = 'Total polarisation fraction'
	data.at[i, t], data.at[i, 'Total polarisation fraction error'] = output[t]

	# DM_obs
	data.at[i, 'DM_obs (pc cm^-3)'] = frb_data.dm

	# YMW16 (requires observed frequency)
	glon, glat = data.at[i, 'Galactic lon'], data.at[i, 'Galactic lat']
	dm, sc = dist_to_dm(glon, glat, globalpars.MW_DIAMETER, nu=freq)
	
	data.at[i, 'DM_MW (YMW16)'], data.at[i, 'Tau_SC (ms) (YMW16)'] = dm.value, sc.value * 1e3 # ms  
	data.at[i, 'DM_MW error (YMW16)'], data.at[i, 'Tau_SC error (YMW16)'] = get_error(glon, glat, nu=freq, method='YMW16')

	# SC_ex
	data.at[i, 'SC_ex'] = (data.at[i, 'Scattering timescale (ms)'] - data.at[i, 'Tau_SC (ms) (YMW16)']) ** 0.5
	data.at[i, 'SC_ex error'] = 0.5 * np.hypot(data.at[i, 'Scattering timescale (ms)'], data.at[i, 'Tau_SC (ms) (YMW16)']) / data.at[i, 'SC_ex']

	# DM_ex
	data.at[i, 'DM_ex (NE2001)'] = data.at[i, 'DM_obs (pc cm^-3)'] - data.at[i, 'DM_MW (NE2001)'] - data.at[i, 'DM_IGM']
	data.at[i, 'DM_ex error (NE2001)'] = np.hypot(data.at[i, 'DM_MW error (NE2001)'], data.at[i, 'DM_IGM error'])


def update_table(file):
	'''
	Update table by calculating host properties and burst properties.
	'''
	igm_state = parameters.State()

	# complete missing host properties
	data = pd.read_csv(file, index_col=0)
	data = data.apply(complete_row, axis=1, state=igm_state)

	# iterate through available FRBs and fill in burst properties
	for entry in []: # get_files('output'): # CHANGE
		with open(entry, 'r') as f:
			# output of curve fit
			output = json.load(f)
			output = output['data'][output['optimum']]

			# input pkl file
			frb_name = get_frb(entry)[:-4]
			frb_base = int(frb_name[:6])
			frb_data = get_data(f'data/{frb_name}.pkl')

			if frb_base in data['FRB'].values:
				complete_burst_properties(frb_base, data, output, frb_data)

	data.to_csv(file)


if __name__ == '__main__':
	# load_hutschenreuter2020()

	file = 'table_new.csv'
	update_table(file)
