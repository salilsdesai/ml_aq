SNAKE_CASE = lambda s: s.lower().replace(' ', '_')

def gather_individual_met_data():
	fields = [
		'Year',
		'Month',
		'Day',
		'Julian Day',
		'Hour',
		'Measurement Height',
		'Indicator Flag',
		'Wind Direction',
		'Wind Speed',
		'Temperature',
		'Lateral Wind Direction Standard Deviation',
		'Vertical Wind Speed Standard Deviation',
		'Sensible Heat Flux',
		'Surface Frication Velocity',
		'Convective Velocity Scale',
		'Potential Temperature Gradient Above the Mixing Height',
		'Convectively Driven Mixing Height',
		'Mechanically Driven Mixing Height',
		'Monin Obukhov Length',
		'Surface Roughness Length',
		'Bowen Ratio',
		'Albedo',
		'Wind Speed Stage 3',
		'Wind Direction Stage 3',
		'Anemometer Height Stage 3',
		'Temperature Stage 3',
		'Measurement Height Stage 3',
		'Precipitation Type',
		'Precipitation Amount',
		'Relative Humidity',
		'Station Pressure',
		'Cloud Cover',
		'Wind Speed Adjustment and Data Source Flag',
		'Cloud Cover and Temperature Substition by Interpolation',
	]
	
	for station_name in ['JFK', 'LGA', 'NYC']:
		
		pfl_file = open('data/ML_AQ/met/' + station_name + '-1317.PFL.PFL', 'r')
		sfc_file = open('data/ML_AQ/met/' + station_name + '-1317.SFC.SFC', 'r')
		pfl = iter(pfl_file)
		sfc = iter(sfc_file)
		next(sfc)

		out_file = open('data/' + station_name.lower() + '_met_data.csv', 'w')

		out_file.write(SNAKE_CASE(fields[0]))
		for i in range(1, len(fields)):
			out_file.write(',' + SNAKE_CASE(fields[i]))

		stop_iteration = False
		while not stop_iteration:
			try:
				pfl_items = next(pfl).strip('\n\r').split()
				sfc_items = next(sfc).strip('\n\r').split()
				vals = sfc_items[0:5] + pfl_items[4:] + sfc_items[5:]

				out_file.write('\n')
				out_file.write(vals[0])
				for i in range(1, len(vals)):
					out_file.write(',' + str(vals[i]))
			except StopIteration:
				stop_iteration = True
		
		pfl_file.close()
		sfc_file.close()
		out_file.close()

def gather_average_met_data():
	fields = [
		None,
		None,
		None,
		None,
		None,
		None,
		'Wind Direction',
		'Wind Speed',
		'Temperature',
		None,
		None,
		None,
		None,
		None,
		None,
		None,
		'Sensible Heat Flux',
		'Surface Frication Velocity',
		'Convective Velocity Scale',
		'Potential Temperature Gradient Above the Mixing Height',
		'Convectively Driven Mixing Height',
		'Mechanically Driven Mixing Height',
		'Monin Obukhov Length',
		'Surface Roughness Length',
		'Bowen Ratio',
		'Albedo',
		'Wind Speed Stage 3',
		'Wind Direction Stage 3',
		None,
		'Temperature Stage 3',
		None,
		None,
		'Precipitation Amount',
		'Relative Humidity',
		'Station Pressure',
		'Cloud Cover',
		None,
		None,
	]

	empty_values = set([9999, 999, 99, 9, -9, -99, -999, -9999])
	stations = [(2, 'NYC'), (3, 'LGA'), (4, 'JFK')]
	out_file = open('data/met_data.csv', 'w')

	out_file.write('id,name,latitude,longitude')
	for f in fields:
		if f is not None:
			out_file.write(',' + f)

	for index in range(len(stations)):

		station_id, station_name = stations[index]

		values_sums = [0] * len(fields)
		values_counts = [0] * len(fields)
		
		pfl_file = open('data/ML_AQ/met/' + station_name + '-1317.PFL.PFL', 'r')
		sfc_file = open('data/ML_AQ/met/' + station_name + '-1317.SFC.SFC', 'r')
		pfl = iter(pfl_file)
		sfc = iter(sfc_file)

		header = next(sfc).split()
		latitude = header[0][:-1]
		longitude = header[1][:-1]

		stop_iteration = False
		while not stop_iteration:
			try:
				pfl_items = next(pfl).strip('\n\r').split()
				sfc_items = next(sfc).strip('\n\r').split()
				vals = pfl_items + sfc_items

				for i in range(len(vals)):
					if fields[i] is not None and float(vals[i]) not in empty_values:
						values_sums[i] += float(vals[i])
						values_counts[i] += 1
			except StopIteration:
				stop_iteration = True
		
		pfl_file.close()
		sfc_file.close()

		out_file.write('\n' + str(station_id) + ',' + station_name + ',' + latitude + ',' + longitude)
		for i in range(len(fields)):
			if fields[i] is not None:
				out_file.write(',' + str(values_sums[i]/values_counts[i]))

	out_file.close()

if __name__ == "__main__":
	gather_individual_met_data()
	gather_average_met_data()

	