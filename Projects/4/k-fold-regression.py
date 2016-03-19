
# Que 4 Part A- Linear Regression with 10-fold validation
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


window_size = 1

for f in file_list:

    print("\n K fold Linear Regression with best 6 features on file  : ", f)
    print("\n   window size : ", window_size)

    f = path + f

    training_data = utility.generate_training_data(f, 1, False, 0, 0, True)

    training_data.pop(0)
    #print training_data

    X = utility.get_feature_matrix(training_data, window_size)

    #print "matrix"
    #print X
    # numpy.roll( ) is used for circular shifting of elements
    train_label = X[2:, 0]
    train_features = X[1:- 1, [0,1,2,6,10,11]]

    model = linear_model.LinearRegression()
    #model = neural_network.MLPRegressor([10, 40, 5], 'relu', 'adam', 0.0001, 200, 'constant', 0.001, 0.5, 200,
    #                                    True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08)


    #predicted_tweet_count = cross_validation.cross_val_predict(model, train_features, train_label, 10, 1, 0, None, 0)
    #print ("predicted_tweet_counts : ", predicted_tweet_count)

    # used mean_absolute_error function
    # returns avg diff between predicted and actual tweets, over 10 folds in a window
    # doesnt randomize the input
    scores = cross_validation.cross_val_score(model, train_features, train_label,  cv=10, scoring='mean_absolute_error')
    avg_scores = numpy.average((-scores))
    print("Average Prediciton Error  : ", avg_scores)

#     hashtag_rmse += avg_scores
#
# # Avg error over all windows of a file
# avg_error = float(hashtag_rmse) / float(window_count)
# print("hashtag avg absolute error for file : "+ f +" ", avg_error)





