#From http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy
from sklearn import linear_model, cross_validation

import matplotlib.pyplot as plt



def day_to_number(str):
    map = {'Monday': 0.0, 'Tuesday': 1.0, 'Wednesday': 2.0, 'Thursday': 3.0, 'Friday': 4.0, 'Saturday': 5.0, 'Sunday': 6.0}
    return map[str]


def number_from_end_string(str):
    return [float(s) for s in str.split('_') if s.isdigit()][-1]


def gen_test_set(data_x, data_y, size):
    data_x_test = []
    data_x_train = []
    data_y_test = []
    data_y_train = []
    for i in range(0, int(data_x.shape[0]/size)):
        rand_pos = numpy.random.random_integers(0, size-1)
        data_x_train.extend(data_x[i:i+rand_pos])
        data_x_test.append(data_x[i+rand_pos])
        data_x_train.extend(data_x[i+rand_pos+1:i+size])
        data_y_train.extend(data_y[i:i+rand_pos])
        data_y_test.append(data_y[i+rand_pos])
        data_y_train.extend(data_y[i+rand_pos+1:i+size])
    data_x_train.extend(data_x[-int(data_x.shape[0] % size):])
    data_y_train.extend(data_y[-int(data_x.shape[0] % size):])
    return data_x_train, data_x_test, data_y_train, data_y_test


def get_selected(all, selected):
    result = []
    for i in selected:
        result.append(all[i])
    return result

network_file = numpy.genfromtxt('../../Datasets/network_backup_dataset.csv',
                                delimiter=',', skip_header=1,
                                converters={1: day_to_number, 3: number_from_end_string, 4: number_from_end_string})
network_X = network_file[:, (0, 1, 2, 3, 4, 6)]
network_Y = network_file[:, 5]

kf = cross_validation.KFold(len(network_X), 10, True)

coefficient_matrix = []
rmse = []
score = []

for train_index, test_index in kf:
    network_X_train = get_selected(network_X, train_index)
    network_X_test = get_selected(network_X, test_index)
    network_Y_train = get_selected(network_Y, train_index)
    network_Y_test = get_selected(network_Y, test_index)

    #network_X_train, network_X_test, network_Y_train, network_Y_test = gen_test_set(network_X, network_Y, 10)

    regr = linear_model.LinearRegression()
    regr.fit(network_X_train, network_Y_train)

    coefficient_matrix.append(regr.coef_)
    rmse.append(numpy.mean((regr.predict(network_X_test) - network_Y_test) ** 2))
    score.append(regr.score(network_X_test, network_Y_test))

print 'Coefficients: \n', coefficient_matrix
print 'RMSE: \n', rmse
print 'Score: \n', score

# # Plot outputs
# #plt.scatter(network_Y_test, network_Y_test,  color='black')
# plt.scatter(network_Y_test, regr.predict(network_X_test), color='blue')
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()