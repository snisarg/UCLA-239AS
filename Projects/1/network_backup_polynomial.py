#From http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy
from sklearn import linear_model, cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


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
network_X_old = network_file[:, (0, 1, 2, 3, 4, 6)]
network_Y = network_file[:, 5]
fixed_set_RMSE = []
average_RMSE = []

for poly_degree in range(1, 8):
    poly = PolynomialFeatures(degree=poly_degree)
    network_X = poly.fit_transform(network_X_old)

    kf = cross_validation.KFold(len(network_X), 10, True)

    coefficient_matrix = []
    rmse = []
    score = []
    predicted = []
    
    rmseFinal = []
    scoreFinal = []
    predictedFinal = []
    for i in range(len(network_X)):
        predicted.append(0)
    
    regr = linear_model.LinearRegression()
    
    for train_index, test_index in kf:
        network_X_train = get_selected(network_X, train_index)
        network_X_test = get_selected(network_X, test_index)
        network_Y_train = get_selected(network_Y, train_index)
        network_Y_test = get_selected(network_Y, test_index)

        regr.fit(network_X_train, network_Y_train)

        coefficient_matrix.append(regr.coef_)
        predicted_values = regr.predict(network_X_test)
        i = 0
        for index in test_index:
            predicted[index] = predicted_values[i]
            i += 1

        rmse.append(numpy.sqrt(((predicted_values - network_Y_test) ** 2).mean()))
        score.append(regr.score(network_X_test, network_Y_test))
    
    #regr.fit(network_X, network_Y)
    predictedFinal = regr.predict(network_X)
    rmseFinal.append(numpy.sqrt(((predictedFinal - network_Y) ** 2).mean()))
    fixed_set_RMSE.append(rmse[0])
    average_RMSE.append(rmseFinal[0])
    print 'RMSE: \n', rmseFinal
    #print 'Coefficients: \n', coefficient_matrix
    #print '-------------\nPolynomial Degree: ', poly_degree
    #print 'RMSE: \n', rmse
    #print 'Score: \n', score

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
# plt.xticks(())
# plt.yticks(())

plt.show()

