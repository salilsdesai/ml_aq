import pandas as pd

SNAKE_CASE = lambda s: s.lower().replace(' ', '_')
MET_STATIONS = ['NYC', 'LGA', 'JFK']
MET_DATA_FIELDS = [
	'Year', 'Month', 'Day', 'Julian Day', 'Hour', 'Measurement Height',
	'Indicator Flag', 'Wind Direction', 'Wind Speed', 'Temperature',
	'Lateral Wind Direction Standard Deviation',
	'Vertical Wind Speed Standard Deviation', 'Sensible Heat Flux',
	'Surface Friction Velocity', 'Convective Velocity Scale',
	'Potential Temperature Gradient Above the Mixing Height',
	'Convectively Driven Mixing Height', 'Mechanically Driven Mixing Height',
	'Monin Obukhov Length', 'Surface Roughness Length', 'Bowen Ratio',
	'Albedo', 'Wind Speed Stage 3', 'Wind Direction Stage 3',
	'Anemometer Height Stage 3', 'Temperature Stage 3', 'Measurement Height Stage 3',
	'Precipitation Type', 'Precipitation Amount', 'Relative Humidity',
	'Station Pressure', 'Cloud Cover', 'Wind Speed Adjustment and Data Source Flag',
	'Cloud Cover and Temperature Substitution by Interpolation',
]
LINK_FIELDS = [
	'ID', 'X', 'Y', 'Nearest Met Station ID',
	'Nearest Met Station Distance',
	'Nearest Met Station Angle', 'Elevation Min', 
	'Elevation Max', 'Elevation Mean', 
	'Traffic Speed', 'Traffic Flow', 'Link Length',
	'Fleet Mix Light', 'Fleet Mix Medium',
	'Fleet Mix Heavy', 'Fleet Mix Commercial',
	'Fleet Mix Bus',
]

def gather_individual_met_data():
	for station_name in MET_STATIONS:
		pfl_file = open('data/ML_AQ/met/' + station_name + '-1317.PFL.PFL', 'r')
		sfc_file = open('data/ML_AQ/met/' + station_name + '-1317.SFC.SFC', 'r')
		pfl = iter(pfl_file)
		sfc = iter(sfc_file)
		next(sfc)

		out_file = open('data/met_data_' + station_name.lower() + '.csv', 'w')

		out_file.write(SNAKE_CASE(MET_DATA_FIELDS[0]))
		for i in range(1, len(MET_DATA_FIELDS)):
			out_file.write(',' + SNAKE_CASE(MET_DATA_FIELDS[i]))

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
	default_values = [
		None, None, None, None, None, None, None, 999, 999, 99.9, None, None, 
		-999, -9, -9, -9, -999, -999, -99999, -9, -9, -9, 999, 999,
		None, 999, None, 9999, -9, 999, 99999, 99, None, None
	]

	out_file = open('data/met_data.csv', 'w')

	out_file.write('id,name,latitude,longitude')
	for i in range(len(MET_DATA_FIELDS)):
		if default_values[i] is not None:
			out_file.write(',' + MET_DATA_FIELDS[i])

	for index in range(len(MET_STATIONS)):
		station_name = MET_STATIONS[index]
		station_id = index + 2
		values_sums = [0] * len(MET_DATA_FIELDS)
		values_counts = [0] * len(MET_DATA_FIELDS)
		
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
				vals = sfc_items[0:5] + pfl_items[4:] + sfc_items[5:]
				for i in range(len(vals)):
					if default_values[i] is not None and (float(vals[i]) != default_values[i]):
						values_sums[i] += float(vals[i])
						values_counts[i] += 1
			except StopIteration:
				stop_iteration = True
		
		pfl_file.close()
		sfc_file.close()

		out_file.write('\n' + str(station_id) + ',' + station_name + ',' + latitude + ',' + longitude)
		for i in range(len(MET_DATA_FIELDS)):
			if default_values[i] is not None:
				out_file.write(',' + str(values_sums[i]/values_counts[i]))
	out_file.close()

def gather_link_data():
	links = {}

	df = pd.read_csv('data/ML_AQ/Met_1_1.csv')
	for row in df.iterrows():
		link = [-1] * len(LINK_FIELDS)
		entry = row[1]
		link[0] = int(entry['ID'])
		link[1] = float(entry['Center_X'])
		link[2] = float(entry['Center_Y'])
		link[3]	= int(entry['NEAR_FID'])
		link[4] = float(entry['NEAR_DIST'])
		link[5] = float(entry['NEAR_ANGLE'])
		links[link[0]] = link
	
	df = pd.read_excel('data/ML_AQ/link_ele_2.xlsx')
	for row in df.iterrows():
		entry = row[1]
		link = links[int(entry['ID'])]
		link[6] = float(entry['MIN'])
		link[7] = float(entry['MAX'])
		link[8] = float(entry['MEAN'])
	
	df = pd.read_csv('data/ML_AQ/Base File.csv')
	
	base_data = {}
	for row in df.iterrows():
		entry = row[1]
		id = int(entry['LinkID'])
		if id in links:
			if id in base_data:
				data = base_data[id]
			else:
				data = ([], [], float(entry['linkLength']))
				base_data[id] = data
			l = data[0] if int(entry['DirectionID']) == -1 else data[1]
			l.append((float(entry['Speed']), float(entry['Flow'])))

	for (id, data) in base_data.items():
		link = links[id]
		# Get the average speed and flow for each direction
		speeds_and_flows = [[0, 0], [0, 0]]
		for i in [0, 1]:
			for entry in data[i]:
				for j in [0, 1]:
					speeds_and_flows[i][j] += (entry[j] / len(data[i]))

		# Speed is weighted average of average speeds in both directions (weighted by flow)
		link[9] = \
			(speeds_and_flows[0][0] * speeds_and_flows [0][1] + speeds_and_flows[1][0] * speeds_and_flows [1][1]) / \
			(speeds_and_flows[0][1] + speeds_and_flows[1][1]) 

		# Flow is sum of average flow in both directions
		link[10] = speeds_and_flows[0][1] + speeds_and_flows[1][1]
		link[11] = data[2] 
	
	df = pd.read_excel('data/ML_AQ/fleet_share.xlsx')

	vehicle_types = ['Light', 'Medium', 'Heavy', 'Commercial', 'Bus']
	times_of_day = [('Morning', 4), ('Midday', 6), ('PM', 4), ('ND', 10)]

	for row in df.iterrows():
		entry = row[1]
		link = links[entry['ID']]
		for i in range(len(vehicle_types)):
			for (time, length) in times_of_day:
				link[i + 12] += (entry[time + '_' + vehicle_types[i] + '_%'] * length / 24)
	
	out_file = open('link_data.csv', 'w')
	format = lambda l: str(l).replace('\'', '').replace(', ', ',')[1:-1] + '\n'
	out_file.write(format(list(map(lambda s: SNAKE_CASE(s), LINK_FIELDS))))
	for link in links.values():
		out_file.write(format(link))
	out_file.close()


if __name__ == "__main__":
	gather_link_data()
