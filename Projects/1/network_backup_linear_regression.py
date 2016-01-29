#From http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt


def day_to_number(str):
    map = {'Monday': 0.0, 'Tuesday': 1.0, 'Wednesday': 2.0, 'Thursday': 3.0, 'Friday': 4.0, 'Saturday': 5.0, 'Sunday': 6.0}
    return map[str]


def number_from_end_string(str):
    return [float(s) for s in str.split('_') if s.isdigit()][-1]

network_file = numpy.genfromtxt('../../Datasets/network_backup_dataset.csv',
                                delimiter=',', skip_header=1,
                                converters={1: day_to_number, 3: number_from_end_string, 4: number_from_end_string})

#print network_file[:30]

network_X = network_file[:, (0, 1, 2, 3, 4, 6)]

network_Y = network_file[:, 5]

network_X_train = network_X[:-3600]
network_X_test = network_X[-3600:]

network_Y_train = network_Y[:-3600]
network_Y_test = network_Y[-3600:]

regr = linear_model.LinearRegression()
regr.fit(network_X_train, network_Y_train)

print 'Coefficients: \n', regr.coef_

print "Residual sum of squares: %.5f" % numpy.mean((regr.predict(network_X_test) - network_Y_test) ** 2)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.5f' % regr.score(network_X_test, network_Y_test))

# Plot outputs
#plt.scatter(network_Y_test, network_Y_test,  color='black')
plt.scatter(network_Y_test, regr.predict(network_X_test), color='blue')

plt.xticks(())
plt.yticks(())

plt.show()