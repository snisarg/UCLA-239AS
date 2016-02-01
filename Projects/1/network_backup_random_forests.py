import numpy
from sklearn import linear_model, cross_validation
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


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

regr = RandomForestRegressor(n_estimators=50,
                          max_features=6,
                          max_depth= 9,
                          n_jobs=1)

for train_index, test_index in kf:
    network_X_train = get_selected(network_X, train_index)
    network_X_test = get_selected(network_X, test_index)
    network_Y_train = get_selected(network_Y, train_index)
    network_Y_test = get_selected(network_Y, test_index)

    # n_estimators - no of trees in the forest
    # max_depth - depth of each tree

    regr.fit(network_X_train,network_Y_train)

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
scoreFinal.append(regr.score(network_X, network_Y))

print 'RMSE: \n', rmseFinal
#print 'Score: \n', scoreFinal

#print 'RMSE: \n', rmse
#print 'Score: \n', score

print 'Coefficients: \n', regr.feature_importances_

#Residual
residual = []
for i in range(len(network_X)):
    residual.append(network_Y[i] - predicted[i])

# Plot outputs
plt.scatter(range(len(network_X)), network_Y,  color='black')
plt.scatter(range(len(network_X)), predicted, color='blue')
plt.plot(range(len(network_X)), residual, color='red', linewidth=1)

# plt.xticks(())
# plt.yticks(())

plt.show()
