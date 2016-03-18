# Que 4 Part B
# Use different features as used in que 3

import os
import numpy
import utility
from sklearn import linear_model, cross_validation, metrics

path = "../../Datasets/tweets/tweet_data/"
test_path = "../../Datasets/tweets/test_data/{}.txt"
#path = "F:/tweets/"

file_list = []
file_list = os.listdir(path)
#file_list = ["subset.txt"]

'''
1. Before Feb. 1, 8:00 a.m.
2. Between Feb. 1, 8:00 a.m. and 8:00 p.m.
3. After Feb. 1, 8:00 p.m.
'''
epoch_8am = 1422806400
epoch_8pm = 1422849600

window_size = 10 # should be 1 for this part of que, clarify
test_files = [['sample1_period1', 'sample4_period1', 'sample5_period1', 'sample8_period1'],
              ['sample2_period2', 'sample6_period2', 'sample9_period2'],
              ['sample3_period3', 'sample7_period3', 'sample10_period3']]

for f in file_list:

    print("Hashtag File ", f)

    f = path + f
    training_data = []

    for i in range(3):

        if i == 0:
            print "Regression on data before Feb 1 8 am"
            training_data = utility.generate_training_data(f, 1, True, 0, epoch_8am, True)
        elif i == 1:
            print "Regression on data between Feb 8 am to 8 pm"
            training_data = utility.generate_training_data(f, 1, True, epoch_8am, epoch_8pm, True)
        else:
            print "Regression on data after Feb 8 pm"
            training_data = utility.generate_training_data(f, 1, True, epoch_8pm, 0, True)

        if len(training_data) != 0:
            training_data.pop(0)

        X = numpy.matrix(training_data)
        rows = X.shape[0]
        avg_error = 0.0
        hashtag_rmse = 0.0
        #generate data for window & perform 10-fold cross-validation and regression

        # Generate training and test data from current window

        data_labels = X[1:, 0]
        data_features = X[:-1, :]

        model = linear_model.LinearRegression()

        # predicted_tweet_count = cross_validation.cross_val_predict(model, data_features, data_labels, 10, 1, 0, None, 0)
        #print ("predicted_tweet_counts : ", predicted_tweet_count)

        # used mean_absolute_error function
        # returns avg diff between predicted and actual tweets, over 10 folds in a window
        # doesnt randomize the input
        scores = cross_validation.cross_val_score(model, data_features, data_labels,  cv=5, scoring='mean_absolute_error')
        avg_scores = numpy.average(-scores)
        print("4th Part A Avg Prediction error for Hashtag file : " + f + " for time frame ", (i+1), avg_scores)

        # FOR QUESTION 5
        model2 = linear_model.LinearRegression()
        model2.fit(data_features, data_labels)

        for file_name in test_files[i]:
            X = utility.generate_training_data(test_path.format(file_name), 1, False, 0, 0, True)
            X = numpy.matrix(X)

            data_labels = X[1:, 0]
            data_features = X[:-1, :]

            predicted = model2.predict(data_features)

            rmse = numpy.sqrt(metrics.mean_squared_error(data_labels, predicted))
            print("For {}, RMSE = {} and Predicted Value = {}".format(file_name, rmse, model2.predict(data_features[-1])))
