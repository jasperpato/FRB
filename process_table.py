from astropy.coordinates import SkyCoord, FK5, Galactic
from pyne2001 import get_galactic_dm
from zdm import parameters, pcosmic
import numpy as np
from pygedm import dist_to_dm
import pandas as pd
import globalpars


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


def get_error(lon, lat, method='NE2001'):
	return 0 if method == 'NE2001' else (0, 0)


def update_table(file):
	data = pd.read_csv(file, dtype=float, index_col=0)
	data['FRB'] = data['FRB'].astype(int)

	igm_state = parameters.State()
	
	for i in data.index:
		# convert to galactic coords
		if not np.isnan(data.at[i, 'Host RA (deg)']) and not np.isnan(data.at[i, 'Host DEC (deg)']):
			data.at[i, 'Galactic longitude'], data.at[i, 'Galactic latitude'] = to_galactic(
				data.at[i, 'Host RA (deg)'],
				data.at[i, 'Host DEC (deg)']
			)
			
			# DM MWs
			data.at[i, 'DM_MW (NE2001)'] = get_galactic_dm(
				data.at[i, 'Galactic longitude'],
				data.at[i, 'Galactic latitude']
			)

			data.at[i, 'DM_MW error (NE2001)'] = get_error(
				data.at[i, 'Galactic longitude'],
				data.at[i, 'Galactic latitude'],
				method='NE2001'
			)

			dm, sc = dist_to_dm(
				data.at[i, 'Galactic longitude'],
				data.at[i, 'Galactic latitude'],
				globalpars.MW_DIAMETER
			)
			data.at[i, 'DM_MW (YMW16)'], data.at[i, 'Tau_SC (s)'] = dm.value, sc.value

			data.at[i, 'DM_MW error (YMW16)'], data.at[i, 'Tau_SC error'] = get_error(
				data.at[i, 'Galactic longitude'],
				data.at[i, 'Galactic latitude'],
				method='YMW16'
			)

			# print(data.at[i, 'Galactic longitude'], data.at[i, 'Galactic latitude'], data.at[i, 'Tau_SC (s)'])

		# DM IGM
		if not np.isnan(data.at[i, 'Host z']):
			data.at[i, 'DM_IGM'] = pcosmic.get_mean_DM(np.array([data.at[i, 'Host z']]), igm_state)[0]
			data.at[i, 'DM_IGM error'] = 0

		# log stellar mass
		data.at[i, 'Log host mass (M_sun)'] = np.log10(data.at[i, 'Host mass (10^9 M_sun)'] * 10e9)

	data.to_csv(file)

if __name__ == '__main__':
	update_table('table.csv')