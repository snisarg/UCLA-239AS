#From http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy
from sklearn import linear_model, cross_validation, neural_network
import matplotlib.pyplot as plt

def day_to_number(str):
    map = {'Monday': 0.0, 'Tuesday': 1.0, 'Wednesday': 2.0, 'Thursday': 3.0, 'Friday': 4.0, 'Saturday': 5.0, 'Sunday': 6.0}
    return map[str]


def number_from_end_string(str):
    return [float(s) for s in str.split('_') if s.isdigit()][-1]


# Get the rows from the 'selected' index.
def get_selected(all_elem, selected):
    result = []
    for i in selected:
        result.append(all_elem[i])
    return result

network_file = numpy.genfromtxt('../../Datasets/network_backup_dataset.csv',
                                delimiter=',', skip_header=1,
                                converters={1: day_to_number, 3: number_from_end_string, 4: number_from_end_string})
# Split files in X attribute columns and the result column object
network_X = network_file[:, (0, 1, 2, 3, 4, 6)]
network_Y = network_file[:, 5]

kf = cross_validation.KFold(len(network_X), 10, True)

coefficient_matrix = []
rmse = []
score = []
predicted = []

coefficient_matrix_final = []
rmseFinal = []
scoreFinal = []
predictedFinal = []

for i in range(len(network_X)):
    predicted.append(0)

model = neural_network.MLPRegressor(100, 'relu', 'adam', 0.0001, 200, 'constant', 0.001, 0.5, 200, True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08)

for train_index, test_index in kf:  #Iterate over the KFold indexes
    # KFold gives indexes, get rows of these indexes appropriately.
    network_X_train = get_selected(network_X, train_index)
    network_X_test = get_selected(network_X, test_index)
    network_Y_train = get_selected(network_Y, train_index)
    network_Y_test = get_selected(network_Y, test_index)

    model.fit(network_X_train, network_Y_train)     # Train

    predicted_values = model.predict(network_X_test)
    i = 0
    for index in test_index:
        predicted[index] = predicted_values[i]      # Record predicted value at the right index
        i += 1

    rmse.append(numpy.sqrt(((predicted_values - network_Y_test) ** 2).mean()))
    score.append(model.score(network_X_test, network_Y_test))


#model.fit(network_X, network_Y)
predictedFinal = model.predict(network_X)

rmseFinal.append(numpy.sqrt(((predictedFinal - network_Y) ** 2).mean()))
scoreFinal.append(model.score(network_X, network_Y))

#print 'Coefficients: \n', coefficient_matrix
#print 'RMSE: \n', rmse
#print 'Score: \n', score

print 'RMSE: \n', rmseFinal
#print 'Score: \n', scoreFinal
print 'Coefficients: \n', model.get_params(True)

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