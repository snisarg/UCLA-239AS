
# Que 4 - Linear Regression with 10-fold validation
# Note: Features used are same as Que 2. But need to use more/other features similar to que 3

import os
import json
import datetime
import numpy
from sklearn import linear_model, neural_network
import utility
from sklearn import linear_model, cross_validation


# Generate training and test data using train_X, train_Y convention for LR
# take no of tweets as train_label & rest of the features as train data or independent vars

path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

file_list = []
file_list = os.listdir(path)
#file_list = ["subset.txt"]


window_size = 11

for f in file_list:

    print("Linear Regression on file : ", f)
    print("window size : ", window_size)

    f = path + f

    training_data = utility.generate_training_data(f, 1, False, 0, 0, False)
    training_data.pop(0)

    X = numpy.matrix(training_data)
    rows = X.shape[0]
    window_count = 0
    avg_error = 0.0
    window_avg_error = 0.0
    hashtag_rmse = 0.0
    #generate data for window & perform 10-fold cross-validation and regression
    window_mean_rmse = 0.0

    # traverse through entire file using sliding windows [1..n], [2..n+1] ...
    for i in range(rows - 1):
        window_end = i + window_size - 1
        if window_end >= rows:
            break
        window_count += 1
        # Generate training and test data from current window

        data_labels = X[i: window_end, 0]
        data_features = X[i: window_end, 1:5]
        '''
        test_label = X[window_end, 0]
        test_features = X[ window_end , [1,4]]
        '''

        #regr = linear_model.LinearRegression()
        # model = linear_model.LinearRegression()
        model = neural_network.MLPRegressor([10, 40, 5], 'relu', 'adam', 0.0001, 200, 'constant', 0.001, 0.5, 200,
                                            True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08)

        # shuffling the window data
        #kf = cross_validation.KFold(window_size, 10, True)

        predicted_tweet_count = cross_validation.cross_val_predict(model, data_features, data_labels, 10, 1, 0, None, 0)
        #print ("predicted_tweet_counts : ", predicted_tweet_count)

        # used mean_absolute_error function
        # returns avg diff between predicted and actual tweets, over 10 folds in a window
        # doesnt randomize the input
        scores = cross_validation.cross_val_score(model, data_features, data_labels,  cv=10, scoring='mean_absolute_error')
        avg_scores = numpy.average(numpy.abs(scores))
        print("avg window error : ", avg_scores)

    #     hashtag_rmse += avg_scores
    #
    # # Avg error over all windows of a file
    # avg_error = float(hashtag_rmse) / float(window_count)
    # print("hashtag avg absolute error for file : "+ f +" ", avg_error)





