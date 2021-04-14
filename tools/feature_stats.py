import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import pi, sin

def extract_list(filepath, class_name=''):
	df = pd.read_csv(filepath)
	return [type(class_name, (object,), dict(row[1])) for row in df.iterrows()]

# Stats Recorder Class by Matt Hancock (http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html)
class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

def get_feature_stats():
    """
    Get mean and std dev for various features
    Used for normalization within model
    """

    links = extract_list('data/link_data.csv', 'link')
    receptors = extract_list('data/receptor_data.csv', 'receptor')
    met_data = {(int(row[1]['id'])):(type('MetStation', (object,), dict(row[1]))) for row in pd.read_csv('data/met_data.csv').iterrows()}

    vmt = lambda link: link.traffic_flow * link.link_length

    features = (
        [
            # ('distance', (lambda link, receptor: (((link.x - receptor.x) ** 2) + ((link.y - receptor.y) ** 2)) ** 0.5)),
            # ('distance_inverse', (lambda link, receptor: (((link.x - receptor.x) ** 2) + ((link.y - receptor.y) ** 2)) ** (-0.5))),
            # ('nearest_link_distance_over_distance', (lambda link, receptor: receptor.nearest_link_distance * ((((link.x - receptor.x) ** 2) + ((link.y - receptor.y) ** 2)) ** (-0.5)))),
            # ('elevation_difference', (lambda link, receptor: receptor.elevation - link.elevation_mean)),
        ],
        [
            ('vmt', vmt),
            ('wind_direction', lambda link: met_data[link.nearest_met_station_id].wind_direction),
            ('wind_speed', lambda link: met_data[link.nearest_met_station_id].wind_speed),
            ('vmt_times_fleet_mix_light', lambda link: vmt(link) * link.fleet_mix_light),
            ('vmt_times_fleet_mix_medium', lambda link: vmt(link) * link.fleet_mix_medium),
            ('vmt_times_fleet_mix_heavy', lambda link: vmt(link) * link.fleet_mix_heavy),
            ('vmt_times_fleet_mix_commercial', lambda link: vmt(link) * link.fleet_mix_commercial),
            ('vmt_times_fleet_mix_bus', lambda link: vmt(link) * link.fleet_mix_bus),
            ('up_down_wind_effect', lambda link: abs(sin((met_data[link.nearest_met_station_id].wind_direction - link.angle) * pi / 180))),
        ],
        [
            'traffic_speed', 'fleet_mix_light', 'fleet_mix_medium',
            'fleet_mix_heavy', 'fleet_mix_commercial', 'fleet_mix_bus', 
            'population_density',
        ],
    )

    srs = [StatsRecorder() for _ in features[0]]

    for link in links:
        for i in range(len(features[0])):
            srs[i].update(np.asarray([[features[0][i][1](link, receptor)] for receptor in receptors]))

    out_file = open('data/feature_stats.csv', 'w')
    
    out_file.write('feature,mean,std_dev\n')

    for i in range(len(features[0])):
        out_file.write(features[0][i][0] + ',' + str(srs[i].mean[0]) + ',' + str(srs[i].std[0]) + '\n')
    
    for (name, f) in features[1]:
        vals = np.asarray([f(link) for link in links])
        out_file.write(name + ',' + str(vals.mean()) + ',' + str(vals.std()) + '\n')

    for attr in features[2]:
        vals = np.asarray([getattr(link, attr) for link in links])
        out_file.write(attr + ',' + str(vals.mean()) + ',' + str(vals.std()) + '\n')

    out_file.close()

def visualize_distribution(filepath, fns):
    df = pd.read_csv(filepath)
    values = [[f(row[1]) for row in df.iterrows()] for f in fns]
    for v in values:
        print(np.mean(v))
        print(np.std(v))
        plt.hist(v)
        plt.show()

if __name__ == "__main__":
    get_feature_stats()