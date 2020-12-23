import numpy as np
from model import extract_list

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

    features = (
        [
            ('distance', (lambda link, receptor: (((link.x - receptor.x) ** 2) + ((link.y - receptor.y) ** 2)) ** 0.5)),
            ('elevation_difference', (lambda link, receptor: receptor.elevation - link.elevation_mean)),
        ],
        [
            ('vmt', (lambda link: link.traffic_flow * link.link_length))
        ],
        [
            'traffic_speed', 'fleet_mix_light', 'fleet_mix_medium',
            'fleet_mix_heavy', 'fleet_mix_commercial', 'fleet_mix_bus',
        ],
    )

    srs = [StatsRecorder() for _ in features[0]]

    for link in links:
        for i in range(len(features[0])):
            srs[i].update(np.asarray([[features[0][i][1](link, receptor)] for receptor in receptors]))

    out_file = open('data/feature_stats.txt', 'w')
    
    for i in range(len(features[0])):
        out_file.write(features[0][i][0] + ',' + str(srs[i].mean[0]) + ',' + str(srs[i].std[0]) + '\n')
    
    for (name, f) in features[1]:
        vals = np.asarray([f(link) for link in links])
        out_file.write(name + ',' + str(vals.mean()) + ',' + str(vals.std()) + '\n')

    for attr in features[2]:
        vals = np.asarray([getattr(link, attr) for link in links])
        out_file.write(attr + ',' + str(vals.mean()) + ',' + str(vals.std()) + '\n')

    out_file.close()

if __name__ == "__main__":
    get_feature_stats()
    