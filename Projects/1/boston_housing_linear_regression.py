#From http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy
from sklearn import linear_model, cross_validation
import matplotlib.pyplot as plt


def get_selected(all_elem, selected):
    result = []
    for i in selected:
        result.append(all_elem[i])
    return result

housing_file = numpy.genfromtxt('../../Datasets/housing_data.csv', delimiter=',', skip_header=1)
housing_X = housing_file[:, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
housing_Y = housing_file[:, 13]

model = linear_model.LinearRegression()

regr = linear_model.LinearRegression()
kf = cross_validation.KFold(len(housing_X), 10, True)
for train_index, test_index in kf:
    network_X_train = get_selected(housing_X, train_index)
    network_Y_train = get_selected(housing_Y, train_index)
    regr.fit(network_X_train, network_Y_train)
    
predicted = cross_validation.cross_val_predict(model, housing_X, housing_Y, 10, 1, 0, None, 0)
scores = cross_validation.cross_val_score(model, housing_X, housing_Y,  cv=10, scoring='mean_squared_error')

print 'All RMSEs',  numpy.sqrt(-scores)
print 'Mean RMSE',  numpy.mean(numpy.sqrt(-scores))
print 'Best RMSE',  numpy.min(numpy.sqrt(-scores))
print 'Coefficients', regr.coef_

#Residual
residual = []
for i in range(len(housing_X)):
    residual.append(housing_Y[i] - predicted[i])

# Plot outputs
sp1 = plt.scatter(range(len(housing_X)), housing_Y,  color='black')
sp2 = plt.scatter(range(len(housing_X)), predicted, color='blue')
#plt.legend((sp1[0], sp2[0]), ('Original values', 'Predicted values'))
#plt.scatter(residual, predicted, color='red')

plt.show()
