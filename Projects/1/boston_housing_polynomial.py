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

housing_file = numpy.genfromtxt('../../Datasets/housing_data.csv', delimiter=',', skip_header=1)
housing_X_old = housing_file[:, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
housing_Y = housing_file[:, 13]

for poly_degree in range(1, 8):
    poly = PolynomialFeatures(degree=poly_degree)
    housing_X = poly.fit_transform(housing_X_old)

    kf = cross_validation.KFold(len(housing_X), 10, True)

    coefficient_matrix = []
    rmse = []
    score = []
    predicted = []
    
    rmseFinal = []
    scoreFinal = []
    predictedFinal = []
    for i in range(len(housing_X)):
        predicted.append(0)

    for train_index, test_index in kf:
        housing_X_train = get_selected(housing_X, train_index)
        housing_X_test = get_selected(housing_X, test_index)
        housing_Y_train = get_selected(housing_Y, train_index)
        housing_Y_test = get_selected(housing_Y, test_index)

        regr = linear_model.LinearRegression()
        regr.fit(housing_X_train, housing_Y_train)

        coefficient_matrix.append(regr.coef_)
        predicted_values = regr.predict(housing_X_test)
        i = 0
        for index in test_index:
            predicted[index] = predicted_values[i]
            i += 1

        rmse.append(numpy.sqrt(((predicted_values - housing_Y_test) ** 2).mean()))
        score.append(regr.score(housing_X_test, housing_Y_test))
    
    regr.fit(housing_X_train, housing_Y_train)
    predictedFinal = regr.predict(housing_X)
    rmseFinal.append(numpy.sqrt(((predictedFinal - housing_Y) ** 2).mean()))
    #print 'Coefficients: \n', coefficient_matrix
    #print '-------------\nPolynomial Degree: ', poly_degree
    #print 'RMSE: \n', rmse
    #print 'Score: \n', score
    
    print 'RMSE: \n', rmseFinal

    #Residual
    residual = []
    for i in range(len(housing_X)):
        residual.append(housing_Y[i] - predicted[i])

    # Plot outputs
    # plt.scatter(range(len(housing_X)), housing_Y,  color='black')
    # plt.scatter(range(len(housing_X)), predicted, color='blue')
    # plt.plot(range(len(housing_X)), residual, color='red', linewidth=1)
    #
    # # plt.xticks(())
    # # plt.yticks(())
    #
    # plt.show()
