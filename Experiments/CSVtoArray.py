import numpy

my_data = numpy.genfromtxt('../Datasets/network_backup_dataset.csv', delimiter=',', skip_header=1, dtype='a')
print my_data[0]

