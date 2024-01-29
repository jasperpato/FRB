'''
Complete the host properties and FRB burst properties in the table CSV file.

Completes data for all FRB pickle files found in /data/pkls that are also present in rows of the table.
'''

from astropy.coordinates import SkyCoord, FK5, Galactic
from pyne2001 import get_galactic_dm
import numpy as np
from pygedm import dist_to_dm
import pandas as pd
import globalpars
from utils import *
import json
from frb.rm import galactic_rm, load_hutschenreuter2020


def split_angle(angle):
	'''
	Split angle string in format H:M:S into components.
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


def get_error(lon, lat, method='NE2001', nu=1, width=2, n_samples=30):
	'''
	Estimate error for DM_MW and Tau_MW by taking the SD of samples within +/- width.
	'''
	lons = np.random.uniform(low=lon - width, high=lon + width, size=n_samples)
	lats = np.random.uniform(low=lat - width, high=lat + width, size=n_samples)

	func = lambda lon, lat: get_galactic_dm(lon, lat) if method == 'NE2001' else (
		np.array([q.value for q in dist_to_dm(lon, lat, globalpars.MW_DIAMETER, nu=nu)])
	)
	ufunc = np.frompyfunc(func, 2, 1)

	return np.std(ufunc(lons, lats))


def get_frb_data(frb_name, threshold=False):
	'''
	Returns FRB pickle data and FRB JSON fit output data, else (None, None) if not found.
	'''
	# frb data
	frb_data = None
	for entry in get_files('data/pkls'):
		if 'pkl' in entry and frb_name in entry:
			frb_data = get_data(entry)

	# output
	frb_output = None
	for entry in get_files('output'):
		if frb_name in entry:
			with open(entry, 'rb') as f:
				frb_output = json.load(f)
				key = 'Threshold R^2' if threshold else 'Max R^2'
				frb_output = frb_output['data'][frb_output[key]]

	return frb_data, frb_output


def complete_row(row, threshold=False):
	'''
	Calculate the missing values in a row of the table.
	'''
	# get frb data and frb output
	frb_name = row['FRB'][2:-1] # get 6-digit frb name
	frb_data, frb_output = get_frb_data(frb_name, threshold)
	
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

	if not frb_data or not frb_output:
		return row
	
	# burst properties
	
	freq = np.mean(frb_data.fmhzarr) * 1e-3 # convert to GHz

	b = b = 'Burst width (ms)'
	row[b] = frb_output[b]

	s = 'Tau_obs (ms)'
	row[s], row['Tau_obs error'] = frb_output[s]

	l = 'Linear polarisation fraction'
	row[l], row[l + ' error'] = frb_output[l]

	t = 'Total polarisation fraction'
	row[t], row[t + ' error'] = frb_output[t]

	# YMW16 (requires observed frequency)
	dm, sc = dist_to_dm(row[glon], row[glat], globalpars.MW_DIAMETER, nu=freq)
	
	row['DM_MW (YMW16)'], row['Tau_MW (ms)'] = dm.value, sc.value * 1e3 # ms  
	row['DM_MW error (YMW16)'], row['Tau_MW error'] = get_error(row[glon], row[glat], nu=freq, method='YMW16')

	return row


def update_table(file, dm_igm_csv, threshold=False):
	'''
	Update table by calculating host properties and burst properties.
	'''
	# complete missing host properties
	data = pd.read_csv(file, index_col=0)
	data = data.apply(complete_row, axis=1, threshold=threshold)

	# DM_IGM
	dm_igm_data = pd.read_csv(dm_igm_csv)
	data['DM_IGM'] = np.interp(data['z'], dm_igm_data['z'], dm_igm_data['mode'])
	data['log(DM_IGM)'] = np.log10(data['DM_IGM'])

	# DM_ex
	data['DM_ex (NE2001)'] = data['DM_obs (pc cm^-3)'] - data['DM_MW (NE2001)'] - data['DM_IGM']
	data['DM_ex error (NE2001)'] = data['DM_MW error (NE2001)'] # no DM_IGM error

	# RM_ex
	data['RM_ex (rad/m^2)'] = data['RM_obs (rad/m^2)'] - data['RM_MW (rad/m^2)']
	data['RM_ex error'] = np.hypot(data['RM_MW error'], data['RM_obs error'])

	# Tau_ex
	data['Tau_ex (ms)'] = (data['Tau_obs (ms)']**2 - data['Tau_MW (ms)']**2) ** 0.5
	data['Tau_ex error'] = 0.5 * np.hypot(data['Tau_obs (ms)'], data['Tau_MW (ms)']) / data['Tau_ex (ms)']
	data['log(Tau_ex)'] = np.log10(data['Tau_ex (ms)'])
	data['log(Tau_ex) error'] = data['Tau_ex error'] / data['Tau_ex (ms)'] / np.log(10)

	# take logs
	data['log(SFR)'] = np.log10(data['SFR 0-100 Myr (Mo yr-1)'])
	data['log(SFR) error'] = data['SFR error'] / data['SFR 0-100 Myr (Mo yr-1)'] / np.log(10)

	data['log(SFR/M)'] = np.log10(data['SFR 0-100 Myr (Mo yr-1)']) - data['log(M*/Mo)']
	data['log(SFR/M) error'] = np.hypot(data['SFR error'] / data['SFR 0-100 Myr (Mo yr-1)'] / np.log(10), data['log(M*/Mo) error'])

	data['log(DM_obs)'] = np.log10(data['DM_obs (pc cm^-3)'])

	data['log(DM_ex)'] = np.log10(data['DM_ex (NE2001)'])
	data['log(DM_ex) error'] = data['DM_ex error (NE2001)'] / data['DM_ex (NE2001)'] / np.log(10)

	data['log(abs(RM_obs))'] = np.log10(np.abs(data['RM_obs (rad/m^2)']))
	data['log(abs(RM_obs)) error'] = np.abs(data['RM_obs error'] / data['RM_obs (rad/m^2)'] / np.log(10))

	data['log(abs(RM_MW))'] = np.log10(np.abs(data['RM_MW (rad/m^2)']))
	data['log(abs(RM_MW)) error'] = np.abs(data['RM_MW error'] / data['RM_MW (rad/m^2)'] / np.log(10))

	data['log(abs(RM_ex))'] = np.log10(np.abs(data['RM_ex (rad/m^2)']))
	data['log(abs(RM_ex)) error'] = np.abs(data['RM_ex error'] / data['RM_ex (rad/m^2)'] / np.log(10))

	data['log(Tau_obs)'] = np.log10(data['Tau_obs (ms)'])
	data['log(Tau_obs) error'] = data['Tau_obs error'] / data['Tau_obs (ms)'] / np.log(10)

	data['log(Tau_MW)'] = np.log10(data['Tau_MW (ms)'])
	data['log(Tau_MW) error'] = data['Tau_MW error'] / data['Tau_MW (ms)'] / np.log(10)

	data['log(abs(Galactic lat))'] = np.log10(np.abs(data['Galactic lat']))
	data['log(1+z)'] = np.log10(1 + np.array(data['z']))

	data.to_csv(file)


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--threshold', action='store_true')
	args = a.parse_args()

	load_hutschenreuter2020()

	file = 'data/table.csv'
	dm_igm_csv = 'data/dm_igm.csv'
	update_table(file, dm_igm_csv, args.threshold)
