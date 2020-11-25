

from dbfread import DBF
from matplotlib import pyplot as plt

RECEPTOR_FILTER = 5
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
                X.append(point[0])
                Y.append(point[1])
            i += 1
        except StopIteration:
            stop_iteration = True
    plt.scatter(X, Y, s=1, color=color)
        
def scatter_receptors():
    scatter(
        iter(DBF('data/receptor_distance.dbf')),
        lambda item: (item['x'], item['y']),
        RECEPTOR_FILTER,
        'red',
    )

def scatter_links():
    file = open('data/Met_1_1.csv', 'r')
    itr = iter(file)
    itr.readline()  # Skip header line
    scatter(
        itr,
        lambda item: ((lambda splits: (float(splits[109]), float(splits[110])))(item.split(','))),
        LINK_FILTER,
        'blue'
    )

scatter_receptors()
scatter_links()
plt.show()
