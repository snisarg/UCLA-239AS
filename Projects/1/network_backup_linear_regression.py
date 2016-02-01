import numpy
from sklearn import linear_model, cross_validation
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

rmse = []
scores = []
predicted = []
coefficient_matrix = []

for i in range(len(network_X)):
    predicted.append(0)

regr = linear_model.LinearRegression()
                    
predicted = cross_validation.cross_val_predict(regr, network_X, network_Y, 10, 1, 0, None, 0)
scores = cross_validation.cross_val_score(regr, network_X, network_Y,  cv=10, scoring='mean_squared_error')

print 'RMSE after CV ',  numpy.sqrt(-scores)

rmse.append(numpy.sqrt(((predicted - network_Y) ** 2).mean()))

print 'RMSE Final ',  rmse

#Residual
residual = []
for i in range(len(network_X)):
    residual.append(network_Y[i] - predicted[i])

# Plot outputs
#plt.scatter(range(len(network_X)), network_Y,  color='black')
#plt.scatter(range(len(network_X)), predicted, color='blue')
#plt.scatter(residual, predicted, color='red')

#plt.show()