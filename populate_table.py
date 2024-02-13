'''
Completes data for all FRB pickle files found in /data/pkls that are also present in rows of the data/table.csv.
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
import os
from luminosity import lum_dist


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


def get_frb_data(frb_name, choose_r2):
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
				frb_output = frb_output['data'][frb_output[choose_r2]]

	return frb_data, frb_output


def complete_host_properties(row, frb_name, path='FRB/frb/data/Galaxies'):
	'''
	Search for host properties JSON in the FRB package and fill in columns.
	'''
	data = {}
	for entry in get_files(path):
		if frb_name in entry:
			with open(entry + f'/FRB{os.path.basename(entry)}_host.json') as f:
				data = json.load(f)
	
	if not data: return

	row['RA'], row['DEC'] = data['ra'], data['dec']
	row['z'] = data['redshift']['z']

	data = data['derived']

	# complete properties (unfinished)


def complete_row(row, choose_r2):
	'''
	Calculate the missing values in a row of the table.
	'''
	# get frb data and frb output
	frb_name = row['FRB'][2:-1] # get 6-digit frb name
	frb_data, frb_output = get_frb_data(frb_name, choose_r2)

	# complete host properties from FRB package
	# complete_host_properties(row, frb_name)
	
	ra, dec = 'RA', 'DEC'
	glon, glat = 'Galactic lon', 'Galactic lat'

	# if not pd.isna(row[ra]) and not pd.isna(row[dec]):
	try:
		# assume RA, DEC are both in hours or both in decimal
		r, d = to_decimal(row[ra], row[dec]) if ':' in row[ra] else (float(row[ra]), float(row[dec]))
		row[glon], row[glat] = to_galactic(r, d)

		# DM NE2001 and error
		row['DM_MW (NE2001)'] = get_galactic_dm(row[glon], row[glat])
		row['DM_MW error (NE2001)'] = get_error(row[glon], row[glat], method='NE2001')

		# RM_MW and error
		coord = SkyCoord(ra=r, dec=d, unit='deg', frame=FK5, equinox='J2000')
		
		rm, rm_error = galactic_rm(coord)
		row['RM_MW (rad/m^2)'], row['RM_MW error'] = rm.value, rm_error.value

	except: pass

	if not frb_data or not frb_output:
		return row
	
	# burst properties
	
	freq = np.mean(frb_data.fmhzarr) * 1e-3 # convert to GHz

	try:
		b = b = 'Burst width (ms)'
		row[b] = frb_output[b]

		s = 'Tau_obs (ms)'
		row[s], row['Tau_obs error'] = frb_output[s]

		if row[s] < row['Tau_obs error']:
			row[s], row['Tau_obs error'] = 0, 0 # replace with zero if Tau_obs is smaller than its error

		l = 'Linear polarisation fraction'
		row[l], row[l + ' error'] = frb_output[l]

		t = 'Total polarisation fraction'
		row[t], row[t + ' error'] = frb_output[t]

		# YMW16 (requires observed frequency)
		if not np.isnan([row[glon], row[glat]]).any():
			dm, sc = dist_to_dm(row[glon], row[glat], globalpars.MW_DIAMETER, nu=freq)
			
			row['DM_MW (YMW16)'], row['Tau_MW (ms)'] = dm.value, sc.value * 1e3 # ms  
			row['DM_MW error (YMW16)'], row['Tau_MW error'] = get_error(row[glon], row[glat], nu=freq, method='YMW16')

	except: pass
	return row


def update_table(file, dm_igm_csv, choose_r2):
	'''
	Update table by calculating host properties and burst properties.
	'''
	# complete missing host properties
	data = pd.read_csv(file, index_col=0)
	data = data.apply(complete_row, axis=1, choose_r2=choose_r2)

	# DM_IGM
	try:
		dm_igm_data = pd.read_csv(dm_igm_csv)
		data['DM_IGM'] = np.interp(data['z'], dm_igm_data['z'], dm_igm_data['mode'])
		data['log(DM_IGM)'] = np.log10(data['DM_IGM'])
	except: pass

	# DM_ex
	try:
		data['DM_ex (NE2001)'] = data['DM_obs (pc cm^-3)'] - data['DM_MW (NE2001)'] - data['DM_IGM']
		data['DM_ex error (NE2001)'] = data['DM_MW error (NE2001)'] # no DM_IGM error
	except: pass

	# RM_ex
	try:
		data['RM_ex (rad/m^2)'] = data['RM_obs (rad/m^2)'] - data['RM_MW (rad/m^2)']
		data['RM_ex error'] = np.hypot(data['RM_MW error'], data['RM_obs error'])
	except: pass

	# Tau_ex
	try:
		data['Tau_ex (ms)'] = (data['Tau_obs (ms)']**2 - data['Tau_MW (ms)']**2) ** 0.5
		data['Tau_ex error'] = ((2 * data['Tau_obs (ms)'] * data['Tau_obs error'])**2 + (2 * data['Tau_MW (ms)'] * data['Tau_MW error'])**2) ** 0.5 / 2 / data['Tau_ex (ms)']
	except: pass

	# Replace Tau_ex with zero if error larger than value
	try:
		cond = data['Tau_ex (ms)'] < data['Tau_ex error']
		data['Tau_ex (ms)'] = np.where(cond, 0, data['Tau_ex (ms)'])
		data['Tau_ex error'] = np.where(cond, 0, data['Tau_ex error'])
	except: pass

	try:
		data['log(Tau_ex)'] = np.log10(data['Tau_ex (ms)'])
		data['log(Tau_ex)'] = np.where(np.isinf(data['log(Tau_ex)']), np.nan, data['log(Tau_ex)']) # replace -inf with np.nan
		data['log(Tau_ex) error'] = data['Tau_ex error'] / data['Tau_ex (ms)'] / np.log(10)
	except: pass

	# take abs and logs
	try:
		data['log(SFR)'] = np.log10(data['SFR 0-100 Myr (Mo yr-1)'])
		data['log(SFR) error'] = data['SFR error'] / data['SFR 0-100 Myr (Mo yr-1)'] / np.log(10)

		data['log(SFR/M)'] = np.log10(data['SFR 0-100 Myr (Mo yr-1)']) - data['log(M*/Mo)']
		data['log(SFR/M) error'] = np.hypot(data['SFR error'] / data['SFR 0-100 Myr (Mo yr-1)'] / np.log(10), data['log(M*/Mo) error'])

		data['log(DM_MW) (NE2001)'] = np.log10(data['DM_MW (NE2001)'])
		data['log(DM_MW) error (NE2001)'] = data['DM_MW error (NE2001)'] / data['DM_MW (NE2001)'] / np.log(10)
		
		data['log(DM_obs)'] = np.log10(data['DM_obs (pc cm^-3)'])

		data['log(DM_ex)'] = np.log10(data['DM_ex (NE2001)'])
		data['log(DM_ex) error'] = data['DM_ex error (NE2001)'] / data['DM_ex (NE2001)'] / np.log(10)

		data['abs(RM_obs)'] = np.abs(data['RM_obs (rad/m^2)'])
		data['abs(RM_obs) error'] = data['RM_obs error']

		data['log(abs(RM_obs))'] = np.log10(data['abs(RM_obs)'])
		data['log(abs(RM_obs)) error'] = np.abs(data['RM_obs error'] / data['RM_obs (rad/m^2)'] / np.log(10))

		data['abs(RM_MW)'] = np.abs(data['RM_MW (rad/m^2)'])
		data['abs(RM_MW) error'] = data['RM_MW error']

		data['log(abs(RM_MW))'] = np.log10(data['abs(RM_MW)'])
		data['log(abs(RM_MW)) error'] = np.abs(data['RM_MW error'] / data['RM_MW (rad/m^2)'] / np.log(10))

		data['abs(RM_ex)'] = np.abs(data['RM_ex (rad/m^2)'])
		data['abs(RM_ex) error'] = data['RM_ex error']

		data['log(abs(RM_ex))'] = np.log10(data['abs(RM_ex)'])
		data['log(abs(RM_ex)) error'] = np.abs(data['RM_ex error'] / data['RM_ex (rad/m^2)'] / np.log(10))

		data['log(Tau_obs)'] = np.log10(data['Tau_obs (ms)'])
		data['log(Tau_obs)'] = np.where(np.isinf(data['log(Tau_obs)']), np.nan, data['log(Tau_obs)'])
		data['log(Tau_obs) error'] = data['Tau_obs error'] / data['Tau_obs (ms)'] / np.log(10)

		data['log(Tau_MW)'] = np.log10(data['Tau_MW (ms)'])
		data['log(Tau_MW) error'] = data['Tau_MW error'] / data['Tau_MW (ms)'] / np.log(10)

		data['abs(Galactic lat)'] = np.abs(data['Galactic lat'])
		data['log(1+z)'] = np.log10(1 + np.array(data['z']))

		data['log(tm)'] = np.log10(data['tm (Gyr)'])
		data['log(tm) error'] = data['tm error'] / data['tm (Gyr)'] / np.log(10)

		# reverse logs
		data['MF/Mo'] = 10 ** data['log(MF/Mo)']
		data['MF/Mo error'] = data['log(MF/Mo) error'] * data['MF/Mo'] * np.log(10)

		data['Z*/Zo'] = 10 ** data['log(Z*/Zo)']
		data['Z*/Zo error'] = data['log(Z*/Zo) error'] * data['Z*/Zo'] * np.log(10)

		data['Zgas/Zo'] = 10 ** data['log(Zgas/Zo)']
		data['Zgas/Zo error'] = data['log(Zgas/Zo) error'] * data['Zgas/Zo'] * np.log(10)

		data['M*/Mo'] = 10 ** data['log(M*/Mo)']
		data['M*/Mo error'] = data['log(M*/Mo) error'] * data['M*/Mo'] * np.log(10)

		data['SFR/M'] = 10 ** data['log(SFR/M)']
		data['SFR/M error'] = data['log(SFR/M) error'] * data['SFR/M'] * np.log(10)

	except: pass

	try:
		# luminosity
		lum_dists2 = np.vectorize(lum_dist)(data['z']) ** 2
		data['Burst energy'] = data['Fluence'] * lum_dists2
		data['Burst energy error'] = data['Fluence error'] * lum_dists2
		data['log(Burst energy)'] = np.log10(data['Burst energy'])
		data['log(Burst energy)'] = data['Burst energy error'] / data['Burst energy'] / np.log(10)
	
	except: pass

	data.to_csv(file)


if __name__ == '__main__':
	from argparse import ArgumentParser

	a = ArgumentParser()
	a.add_argument('--choose-r2', default='No increase R^2')
	a.add_argument('file', default='data/table.csv')
	a.add_argument('--igm-csv', default='data/dm_igm.csv')
	args = a.parse_args()

	load_hutschenreuter2020()

	update_table(args.file, args.igm_csv, choose_r2=args.choose_r2)
