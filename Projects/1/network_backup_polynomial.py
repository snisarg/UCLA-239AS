#From http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy
from sklearn import linear_model, cross_validation
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model.base import LinearRegression


def day_to_number(str):
    map = {'Monday': 0.0, 'Tuesday': 1.0, 'Wednesday': 2.0, 'Thursday': 3.0, 'Friday': 4.0, 'Saturday': 5.0, 'Sunday': 6.0}
    return map[str]


def number_from_end_string(str):
    return [float(s) for s in str.split('_') if s.isdigit()][-1]


def get_selected(all_elem, selected):
    result = []
    for i in selected:
        result.append(all_elem[i])
    return result

network_file = numpy.genfromtxt('../../Datasets/network_backup_dataset.csv',
                                delimiter=',', skip_header=1,
                                converters={1: day_to_number, 3: number_from_end_string, 4: number_from_end_string})
network_X = network_file[:, (0, 1, 2, 3, 4, 6)]
network_Y = network_file[:, 5]

fixed_set_RMSE = []
average_RMSE = []

for poly_degree in range(1, 8):
    regr = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())

    predicted = cross_validation.cross_val_predict(regr, network_X, network_Y, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(regr, network_X, network_Y,  cv=10, scoring='mean_squared_error')
    
    print '----poly_degree---', poly_degree
    print 'All RMSEs',  numpy.sqrt(-scores)
    print 'Mean RMSE',  numpy.mean(numpy.sqrt(-scores))
    print 'Best RMSE',  numpy.min(numpy.sqrt(-scores))
    
    fixed_set_RMSE.append(numpy.mean(numpy.sqrt(-scores[0])))
    average_RMSE.append(numpy.mean(numpy.sqrt(-scores)))
    
    #Residual
    residual = []
    for i in range(len(network_X)):
        residual.append(network_Y[i] - predicted[i])

# Plot outputs
#plt.scatter(range(len(network_X)), network_Y,  color='black')
#plt.scatter(range(len(network_X)), predicted, color='blue')'
print 'fixed_set_rmse: ', fixed_set_RMSE
print 'average: ', average_RMSE
p1 = plt.plot(range(1, 8), fixed_set_RMSE, color='red', linewidth=1)
p2 = plt.plot(range(1, 8), average_RMSE, color='blue', linewidth=1)
plt.xlabel('Polynomial')
plt.ylabel('RMSE')
plt.title('RMSE for varying polynomials.')
plt.grid(True)
plt.legend((p1[0], p2[0]), ('Fixed Set RMSE', 'Average RMSE'))

plt.show()