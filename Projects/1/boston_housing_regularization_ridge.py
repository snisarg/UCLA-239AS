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

kf = cross_validation.KFold(len(housing_X), 10, True)

rmse = []
predicted = []
for i in range(len(housing_X)):
    predicted.append(0)

alphas = [0.1, 0.01, 0.001]
    
reglzn = linear_model.RidgeCV(alphas, False, False, None, kf, None, False)
reglzn.fit(housing_X, housing_Y)

predicted_values = reglzn.predict(housing_X)

rmse.append(numpy.sqrt(((predicted_values - housing_Y) ** 2).mean()))

print 'RMSE: \n', rmse
print 'Alpha: \n', reglzn.alpha_
