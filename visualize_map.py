

from dbfread import DBF
from matplotlib import pyplot as plt

RECEPTOR_FILTER = 10
LINK_FILTER = 1

def scatter(itr, get_point, filter, color):
    X = []
    Y = []
    i = 0
    stop_iteration = False
    while not stop_iteration:
        try:
            item = next(itr)
            if (i % filter == 0):
                point = get_point(item)
                if point is not None:
                    X.append(point[0])
                    Y.append(point[1])
            i += 1
        except StopIteration:
            stop_iteration = True
    plt.scatter(X, Y, s=1, color=color)
        
def scatter_receptors():
    scatter(
        iter(DBF('data/ML_AQ/receptor_distance.dbf')),
        lambda item: (item['x'], item['y']),
        RECEPTOR_FILTER,
        'red',
    )

def scatter_links():
    file = open('data/ML_AQ/Met_1_1.csv', 'r')
    itr = iter(file)
    itr.readline()  # Skip header line
    scatter(
        itr,
        lambda item: ((lambda splits: (float(splits[4]), float(splits[5])) if splits[1] != '246543' else None)(item.split(','))),
        LINK_FILTER,
        'blue'
    )

if __name__ == "__main__":
    scatter_receptors()
    scatter_links()
    plt.show()
