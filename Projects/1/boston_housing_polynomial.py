#From http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy
from sklearn import linear_model, cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def get_selected(all_elem, selected):
    result = []
    for i in selected:
        result.append(all_elem[i])
    return result

network_file = numpy.genfromtxt('../../Datasets/housing_data.csv', delimiter=',', skip_header=1)
network_X_old = network_file[:, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
network_Y = network_file[:, 13]

for poly_degree in range(1, 8):
    poly = PolynomialFeatures(degree=poly_degree)
    network_X = poly.fit_transform(network_X_old)

    kf = cross_validation.KFold(len(network_X), 10, True)

    coefficient_matrix = []
    rmse = []
    score = []
    predicted = []
    for i in range(len(network_X)):
        predicted.append(0)

    for train_index, test_index in kf:
        network_X_train = get_selected(network_X, train_index)
        network_X_test = get_selected(network_X, test_index)
        network_Y_train = get_selected(network_Y, train_index)
        network_Y_test = get_selected(network_Y, test_index)

        regr = linear_model.LinearRegression()
        regr.fit(network_X_train, network_Y_train)

        coefficient_matrix.append(regr.coef_)
        predicted_values = regr.predict(network_X_test)
        i = 0
        for index in test_index:
            predicted[index] = predicted_values[i]
            i += 1

        rmse.append(numpy.mean(predicted_values - network_Y_test) ** 2)
        score.append(regr.score(network_X_test, network_Y_test))

    #print 'Coefficients: \n', coefficient_matrix
    print '-------------\nPolynomial Degree: ', poly_degree
    print 'RMSE: \n', rmse
    print 'Score: \n', score

    #Residual
    residual = []
    for i in range(len(network_X)):
        residual.append(network_Y[i] - predicted[i])

    # Plot outputs
    # plt.scatter(range(len(network_X)), network_Y,  color='black')
    # plt.scatter(range(len(network_X)), predicted, color='blue')
    # plt.plot(range(len(network_X)), residual, color='red', linewidth=1)
    #
    # # plt.xticks(())
    # # plt.yticks(())
    #
    # plt.show()
