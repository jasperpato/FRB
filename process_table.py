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


def get_frb_data(frb_name):
	'''
	Attempt to find frb data and frb output on file, else returns None, None.
	'''
	# frb data
	frb_data = None
	for entry in get_files('data'):
		if 'pkl' in entry and frb_name in entry:
			frb_data = get_data(entry)

	# output
	frb_output = None
	for entry in get_files('output'):
		if frb_name in entry:
			with open(entry, 'rb') as f:
				frb_output = json.load(f)
				frb_output = frb_output['data'][frb_output['optimum']] # get only optimal fit

	return frb_data, frb_output

def complete_row(row, state):
	'''
	Complete the missing columns of a row.
	'''

	# get frb data and frb output
	frb_name = row['FRB'][2:-1] # get 6-digit frb name
	frb_data, frb_output = get_frb_data(frb_name)
	
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

	if not frb_data or not frb_output:
		return row
	
	# burst properties
	
	freq = np.mean(frb_data.fmhzarr) * 1e-3 # convert to GHz

	b = b = 'Burst width (ms)'
	row[b] = frb_output[b]

	s = 'Scattering timescale (ms)'
	row[s], row['Scattering timescale error'] = frb_output[s]

	l = 'Linear polarisation fraction'
	row[l], row[l + ' error'] = frb_output[l]

	t = 'Total polarisation fraction'
	row[t], row[t + ' error'] = frb_output[t]

	# DM_obs
	row['DM_obs (pc cm^-3)'] = frb_data.dm

	# YMW16 (requires observed frequency)
	dm, sc = dist_to_dm(row[glon], row[glat], globalpars.MW_DIAMETER, nu=freq)
	
	row['DM_MW (YMW16)'], row['Tau_SC (ms) (YMW16)'] = dm.value, sc.value * 1e3 # ms  
	row['DM_MW error (YMW16)'], row['Tau_SC error (YMW16)'] = get_error(row[glon], row[glat], nu=freq, method='YMW16')

	# SC_ex
	# print(row[s], row['Tau_SC (ms) (YMW16)'])

	row['SC_ex'] = (row[s] - row['Tau_SC (ms) (YMW16)']) ** 0.5
	row['SC_ex error'] = 0.5 * np.hypot(row[s], row['Tau_SC (ms) (YMW16)']) / row['SC_ex']

	# DM_ex
	row['DM_ex (NE2001)'] = row['DM_obs (pc cm^-3)'] - row['DM_MW (NE2001)'] - row['DM_IGM']
	row['DM_ex error (NE2001)'] = np.hypot(row['DM_MW error (NE2001)'], row['DM_IGM error'])

	return row

def update_table(file):
	'''
	Update table by calculating host properties and burst properties.
	'''
	igm_state = parameters.State()

	# complete missing host properties
	data = pd.read_csv(file, index_col=0)
	data = data.apply(complete_row, axis=1, state=igm_state)

	# reorder columns
	order = [
		'FRB', 'RA', 'DEC', 'Galactic lat', 'Galactic lon', 'z', 'Repeater', 'Host magnitude (AB)', 
		'log(MF/Mo)', 'log(MF/Mo) error', 'log(Z*/Zo)', 'log(Z*/Zo) error', 'Av,old (mag)', 'Av,old error', 'Av,young (mag)', 'Av,young error', 'AGN',
		'log(Zgas/Zo)', 'log(Zgas/Zo) error', 'SFR 0-100 Myr (Mo yr-1)', 'SFR error', 'log(M*/Mo)', 'log(M*/Mo) error', 'tm (Gyr)', 'tm error',
		'DM_MW (NE2001)', 'DM_MW error (NE2001)', 'DM_MW (YMW16)', 'DM_MW error (YMW16)', 'DM_IGM', 'DM_IGM error', 'DM_obs (pc cm^-3)', 'DM_ex (NE2001)', 'DM_ex error (NE2001)',
		'RM_MW (rad/m^2)', 'RM_MW error', 'Tau_SC (ms) (YMW16)', 'Tau_SC error (YMW16)', 'Scattering timescale (ms)', 'Scattering timescale error', 'SC_ex', 'SC_ex error',
		'Linear polarisation fraction', 'Linear polarisation fraction error', 'Total polarisation fraction', 'Total polarisation fraction error',  'Burst width (ms)',        
	]
	data = data[order]

	data.to_csv(file)


if __name__ == '__main__':
	load_hutschenreuter2020()

	file = 'data/table_new.csv'
	update_table(file)
