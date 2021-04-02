import pandas as pd
from math import isnan
from simpledbf import Dbf5
from model import extract_list

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
	'Nearest Met Station Angle', 'Elevation Mean', 
	'Traffic Speed', 'Traffic Flow', 'Link Length',
	'Fleet Mix Light', 'Fleet Mix Medium',
	'Fleet Mix Heavy', 'Fleet Mix Commercial',
	'Fleet Mix Bus', 'Angle',
]
RECEPTOR_FIELDS = [
	'ID', 'X', 'Y', 'Elevation', 'Pollution Concentration', 'Nearest Link Distance'
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
			out_file.write(',' + SNAKE_CASE(MET_DATA_FIELDS[i]))

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

def write_lists_to_csv(name, fields, items):
	out_file = open('data/' + name + '_data.csv', 'w')
	format = lambda l: str(l).replace('\'', '').replace(', ', ',')[1:-1] + '\n'
	out_file.write(format(list(map(lambda s: SNAKE_CASE(s), fields))))
	for item in items.values():
		out_file.write(format(item))
	out_file.close()

def gather_link_data():
	links = {}

	# Get Location and Meteorological Station Data
	df = pd.read_csv('data/ML_AQ/Met_1_1.csv')
	for row in df.iterrows():
		entry = row[1]
		id = int(entry['ID'])
		link = [-1] * len(LINK_FIELDS)
		link[0] = id
		link[1] = float(entry['Center_X'])
		link[2] = float(entry['Center_Y'])
		link[3]	= int(entry['NEAR_FID'])
		link[4] = float(entry['NEAR_DIST'])
		link[5] = float(entry['NEAR_ANGLE'])
		links[link[0]] = link
	
	# Get Elevation Data
	elevations = {}
	for (df, field_name) in [(pd.read_excel('data/ML_AQ/link_ele_2.xlsx'), 'MEAN'), (pd.read_excel('data/ML_AQ/elevation_2_december_18_2020.xlsx'), 'RASTERVALU')]:
		for row in df.iterrows():
			entry = row[1]
			id = int(entry['ID'])
			if id not in elevations:
				elevations[id] = []
			elevations[id].append(float(entry[field_name]))
	
	for (id, all_elevations) in elevations.items():
		links[id][6] = sum(all_elevations)/len(all_elevations)

	links[246543][6] = 0	# TODO: Remove once we have 246543's elevation
	
	# Get Length, Speed, Flow
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
		link[7] = \
			(speeds_and_flows[0][0] * speeds_and_flows [0][1] + speeds_and_flows[1][0] * speeds_and_flows [1][1]) / \
			(speeds_and_flows[0][1] + speeds_and_flows[1][1]) 

		# Flow is sum of average flow in both directions
		link[8] = speeds_and_flows[0][1] + speeds_and_flows[1][1]
		link[9] = data[2] 
	
	# Get Fleet Mix
	df = pd.read_excel('data/ML_AQ/fleet_share.xlsx')
	vehicle_types = ['Light', 'Medium', 'Heavy', 'Commercial', 'Bus']
	times_of_day = [('Morning', 4), ('Midday', 6), ('PM', 4), ('ND', 10)]

	for row in df.iterrows():
		entry = row[1]
		link = links[entry['ID']]
		for i in range(len(vehicle_types)):
			link[i + 10] = 0
			total_hours = 0
			for (time, length) in times_of_day:
				curr_fleet_mix = entry[time + '_' + vehicle_types[i] + '_%']
				if not isnan(curr_fleet_mix):
					link[i + 10] += (curr_fleet_mix * length)
					total_hours += length
			link[i + 10] /= total_hours

	df = pd.read_csv('data/ML_AQ/link_angle_jan19.csv')
	for row in df.iterrows():
		entry = row[1]
		link = links[entry['ID']]
		link[15] = float(entry['angle'])
	
	write_lists_to_csv('link', LINK_FIELDS, links)

def gather_receptor_data():
	receptors = {}
	df = Dbf5('data/ML_AQ/receptor_distance.dbf').to_dataframe()
	for row in df.iterrows():
		entry = row[1]
		conc = float(entry['conc'])
		if conc > 0:
			receptor = [-1] * len(RECEPTOR_FIELDS)
			receptor[0] = int(entry['OBJECTID'])
			receptor[1] = float(entry['x'])
			receptor[2] = float(entry['y'])
			receptor[4] = conc
			receptors[receptor[0]] = receptor
	
	df = pd.read_excel('data/ML_AQ/Receptors_ele_1.xlsx')
	for row in df.iterrows():
		entry = row[1]
		if int(entry['Field1']) in receptors:
			receptors[int(entry['Field1'])][3] = float(entry['Ezlevation (Meter)'])
	
	df = pd.read_csv('data/link_data.csv')
	links = [(row[1]['x'], row[1]['y']) for row in df.iterrows()]
	for receptor in receptors.values():
		receptor[5] = min([((l[0] - receptor[1]) ** 2 + (l[1] - receptor[2]) ** 2) ** (0.5) for l in links])
	
	write_lists_to_csv('receptor', RECEPTOR_FIELDS, receptors)

def gather_link_population_densities():
	# 563312.4167, 4483978.488 → 40.499955, -74.253425
	# 596926.7968, 4528977.4090 → 40.905391, -73.847027
	LON = lambda x: 0.000012090004301462486 * x - 81.06387454097022
	LAT = lambda y: 0.000009009904926387032 * y + 0.0997351311553274

	links = extract_list('data/link_data.csv', 'Link')
	pop = extract_list('data/ML_AQ/ny_population_data.csv', 'PopulationData')
	latlon = extract_list('data/ML_AQ/ny_lat_lon.csv', 'LatLon')

	for l in links:
		closest = None
		closest_dist = None
		lat = LAT(l.y)
		lon = LON(l.x)
		for ll in latlon:
			dist = (((ll.lat - lat) ** 2) + ((ll.lon - lon) ** 2)) ** 0.5
			if (closest_dist is None or dist < closest_dist):
				closest_dist = dist
				closest = ll
		l.zipcode = closest.zipcode
		
	pop_density_map = {p.zipcode:getattr(p, 'population density (per sq mile)') for p in pop}
	for l in links:
		l.population_density = pop_density_map[l.zipcode]

	write_lists_to_csv('link_population_density', ['id', 'population_density'], {l: [l.id, l.population_density] for l in links})

def gather_temporal_data():
	DIRECTORY = ''

	def list_to_csv_line(l):
		return str(l).replace(' ', '').replace("'", '')[1:-1] + '\n'

	times_of_day = [('Morning', 4), ('Midday', 6), ('PM', 4), ('ND', 10)]

	hour_periods = []
	for _ in range(6):
		hour_periods.append(3)
	for _ in range(4):
		hour_periods.append(0)
	for _ in range(6):
		hour_periods.append(1)
	for _ in range(4):
		hour_periods.append(2)
	for _ in range(4):
		hour_periods.append(3)

	# Link Data
	links = {}

	df = pd.read_csv(DIRECTORY + 'data/ML_AQ/Met_1_1.csv')
	for row in df.iterrows():
		entry = row[1]
		id = int(entry['ID'])
		link = [[None for _ in range(7)] for _ in range(4)]
		links[id] = link

	df = pd.read_csv(DIRECTORY + 'data/ML_AQ/Base File.csv')

	# Get Length, Speed, Flow
	base_data = {}
	for row in df.iterrows():
		entry = row[1]
		id = int(entry['LinkID'])
		period = hour_periods[int(entry['HourID']) - 1]	# indexed by 1 for some reason
		if id in links:
			if id in base_data and period in base_data[id]:
				data = base_data[id][period]
			else:
				if id not in base_data:
					base_data[id] = {}
				data = ([], [])
				base_data[id][period] = data
			l = data[0] if int(entry['DirectionID']) == -1 else data[1]
			l.append((float(entry['Speed']), float(entry['Flow'])))

	for (id, data) in base_data.items():
		for period in range(4):
			if period in base_data[id]:
				data = base_data[id][period]
				link = links[id][period]
				# Get the average speed and flow for each direction
				speeds_and_flows = [[0, 0], [0, 0]]
				for i in [0, 1]:
					for entry in data[i]:
						for j in [0, 1]:
							speeds_and_flows[i][j] += (entry[j] / len(data[i]))

				# Speed is weighted average of average speeds in both directions (weighted by flow)
				link[0] = \
					(speeds_and_flows[0][0] * speeds_and_flows[0][1] + speeds_and_flows[1][0] * speeds_and_flows[1][1]) / \
					(speeds_and_flows[0][1] + speeds_and_flows[1][1]) 

				# Flow is sum of average flow in both directions
				link[1] = speeds_and_flows[0][1] + speeds_and_flows[1][1]

	df = pd.read_excel(DIRECTORY + 'data/ML_AQ/fleet_share.xlsx')

	# Get Fleet Mix
	vehicle_types = ['Light', 'Medium', 'Heavy', 'Commercial', 'Bus']

	for row in df.iterrows():
		entry = row[1]
		link = links[entry['ID']]
		for j in range(len(times_of_day)):
			(time, _) = times_of_day[j]
			for i in range(len(vehicle_types)):
				curr_fleet_mix = entry[time + '_' + vehicle_types[i] + '_%']
				if isnan(curr_fleet_mix):
					curr_fleet_mix = 0
				link[j][2 + i] = (curr_fleet_mix)

	# Replace Nones with ones from before
	num_replace = 0
	for link in links.values():
		for i in range(len(link)):
			for j in range(len(link[i])):
				if link[i][j] is None:
					# Need to replace
					k = i - 1
					if k < 0:
						k = len(link) - 1
					while link[k][j] is None and k != i:
						k = k - 1
						if k < 0:
							k = len(link) - 1
					if k == i:
						print('FAILED: ' + str((i, j)))
					else:
						link[i][j] = link[k][j]
						num_replace += 1
	print(num_replace)

	headers = ['traffic_speed', 'traffic_flow']
	for s in vehicle_types:
		headers.append('fleet_mix_' + s.lower())

	csv_headers = ['id']
	for (t, _) in times_of_day:
		for h in headers:
			csv_headers.append(h + '_' + t.lower())

	f = open('link_data_temporal.csv', 'w')
	f.write(list_to_csv_line(csv_headers))
	for (id, l) in links.items():
		data = [id]
		for t in l:
			for q in t:
				data.append(q)
		f.write(list_to_csv_line(data))
	f.close()

	# Met Data
	directions = []
	speeds = []
	stations = [(2, 'nyc'), (3, 'jfk'), (4, 'lga')]
	for (id, name) in stations:
		d = [[0, 0] for _ in range(4)]
		s = [[0, 0] for _ in range(4)]
		df = pd.read_csv(DIRECTORY + 'data/met_data_' + name + '.csv')
		for row in df.iterrows():
			entry = row[1]
			dir = int(entry['wind_direction'])
			spd = float(entry['wind_speed'])
			period = hour_periods[int(entry['hour']) - 1]
			if dir != 999:
				d[period][0] += dir
				d[period][1] += 1
			if spd != 999:
				s[period][0] += spd
				s[period][1] += 1
				
		directions.append([d[i][0]/d[i][1] if d[i][1] != 0 else 0 for i in range(4)])
		speeds.append([s[i][0]/s[i][1] if s[i][1] != 0 else 0 for i in range(4)])

	headers = ['wind_direction', 'wind_speed']

	csv_headers = ['id', 'name']
	for (t, _) in times_of_day:
		for h in headers:
			csv_headers.append(h + '_' + t.lower())

	f = open('met_data_temporal.csv', 'w')
	f.write(list_to_csv_line(csv_headers))
	for i in range(len(stations)):
		id = stations[i][0]
		name = stations[i][1]
		data = [id, name]
		for j in range(4):
			data.append(directions[i][j])
			data.append(speeds[i][j])
		f.write(list_to_csv_line(data))
	f.close()
		

if __name__ == "__main__":
	gather_receptor_data()
