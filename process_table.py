from astropy.coordinates import SkyCoord, FK5, Galactic
from pyne2001 import get_galactic_dm
from zdm import parameters, pcosmic
import numpy as np
from pygedm import dist_to_dm
import pandas as pd
import globalpars
from utils import *
import json


def split_angle(angle):
	'''
	Split angle string into components.
	'''
	angle = angle.replace('h', 'd')
	first, rest = angle.split('d')
	second, third = rest.split('m')
	if third[-1] == 's': third = third[:-1]
	return float(first), float(second), float(third)


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


def get_error(lon, lat, method='NE2001', width=2, size=30):
	'''
	Estimate error for DM and SM by taking size samples within +/- width.
	'''
	lons = np.random.uniform(low=lon - width, high=lon + width, size=size)
	lats = np.random.uniform(low=lat - width, high=lat + width, size=size)

	func = lambda lon, lat: get_galactic_dm(lon, lat) if method == 'NE2001' else (
		np.array([q.value for q in dist_to_dm(lon, lat, globalpars.MW_DIAMETER)])
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
	# transform ra, dec to decimal if necessary
	in_minutes = lambda data: type(data) == str and 'm' in data

	ra, dec = 'Host RA (deg)', 'Host DEC (deg)'
	glon, glat = 'Galactic longitude', 'Galactic latitude'

	if in_minutes(row[ra]) and in_minutes(row[dec]):
		row[ra], row[dec] = to_decimal(row[ra], row[dec])

	# transform ra, dec to galactic coords
	if not np.isnan([row[ra], row[dec]]).any():
		row[glon], row[glat] = to_galactic(row[ra], row[dec])

	if not np.isnan([row[glon], row[glat]]).any():
		# get DM NE2001 and error
		row['DM_MW (NE2001)'] = get_galactic_dm(row[glon], row[glat])
		row['DM_MW error (NE2001)'] = get_error(row[glon], row[glat], method='NE2001')

		#get DM and SM YMW16 and error
		dm, sc = dist_to_dm(row[glon], row[glat], globalpars.MW_DIAMETER)
		row['DM_MW (YMW16)'], row['Tau_SC (ms) (YMW16)'] = dm.value, sc.value * 1e3 # ms  
		row['DM_MW error (YMW16)'], row['Tau_SC error (YMW16)'] = get_error(row[glon], row[glat], method='YMW16')

	# get DM IGM and error
	if not np.isnan(row['Host z']):
		row['DM_IGM'] = pcosmic.get_mean_DM(np.array([row['Host z']]), state)[0]
		row['DM_IGM error'] = get_igm_error(row['Host z'], state)

	# log stellar mass
	row['Log host mass (M_sun)'] = np.log10(row['Host mass (10^9 M_sun)'] * 10e9)

	return row


def complete_burst_properties(frb, data, output, frb_data):
	'''
	Search for FRB in data and fill in burst properties.
	'''
	x = data['FRB'] == frb

	# burst properties
	b = 'Burst width (ms)'
	data.loc[x, b] = output[b]

	s = 'Scattering timescale (ms)'
	data.loc[x, s], data.loc[x, 'Scattering timescale error'] = output[s]

	l = 'Linear polarisation fraction'
	data.loc[x, l], data.loc[x, 'Linear polarisation fraction error'] = output[l]

	t = 'Total polarisation fraction'
	data.loc[x, t], data.loc[x, 'Total polarisation fraction error'] = output[t]

	# DM
	data.loc[x, 'DM_obs'] = frb_data.dm

def update_table(file):
	'''
	Update table by calculating host properties and burst properties.
	'''
	igm_state = parameters.State()

	# complete missing host properties
	data = pd.read_csv(file, index_col=0)
	# data = data.apply(complete_row, axis=1, state=igm_state)
	data['FRB'] = data['FRB'].astype(int)

	# iterate through available FRBs and fill in burst properties
	for entry in get_files('output'):
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
	file = 'table.csv'
	update_table(file)
