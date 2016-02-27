import numpy


def get_R():
    my_data = numpy.genfromtxt('../../Datasets/ml-100k/u.data', delimiter='\t')
    max_users = numpy.max(my_data[:, 0])
    max_movies = numpy.max(my_data[:, 1])
    r = numpy.zeros((max_users, max_movies))
    for row in my_data:
        r[row[0]-1, row[1]-1] = row[2]
    return r

