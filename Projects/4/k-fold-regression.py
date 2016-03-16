import os
import json
import datetime
import numpy
from sklearn import linear_model
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

    print("Linear Regression on file", f)
    print("window size", window_size)

    f = path + f

    training_data = utility.generate_training_data(f, 1)
    training_data.pop(0)

    X = numpy.matrix(training_data)
    rows = X.shape[0]
    window_count = 0
    avg_error = 0.0
    window_avg_error = 0.0
    hashtag_rmse = 0.0
    #generate data for window & perform 10-fold cross-validation and regression
    window_mean_rmse = 0.0


    for i in range(rows - 1):
        window_end = i + window_size - 1
        if window_end >= rows:
            break
        window_count += 1
        # Generate training and test data from current window
        train_label = X[i: window_end, 0]
        train_features = X[i: window_end, [1, 4]]
        test_label = X[window_end, 0]
        test_features = X[ window_end , [1,4]]


        avg_error = 0.0
        error = 0.0

        regr = linear_model.LinearRegression()
        model = linear_model.LinearRegression()
        kf = cross_validation.KFold(rows, 10, True)

        for train_index, test_index in kf:
            X_train, X_test = X[train_index, 1:], X[test_index, 1:]
            Y_train, Y_test = X[train_index, 0], X[test_index, 0]

            regr.fit(X_train, Y_train) # to get co-efficients

        # issues - what data shud be passed to predict - test or whole ?
        predicted = cross_validation.cross_val_predict(model, train_features, train_label, 10, 1, 0, None, 0)
        scores = cross_validation.cross_val_score(model, train_features, train_label,  cv=10, scoring='mean_squared_error')

        window_mean_rmse = numpy.mean(numpy.sqrt(-scores))
        print 'Mean RMSE',  numpy.mean(numpy.sqrt(-scores))
        print 'Best RMSE',  numpy.min(numpy.sqrt(-scores))

            #print("Co-efficients : ", regr.coef_)
            #print("Residual sum of squares: %.2f"% numpy.mean((regr.predict(Y_test) - X_test) ** 2))
            #predicted_tweets = regr.predict(X_test)
            #error += numpy.mean((predicted_tweets - Y_test) ** 2)

        hashtag_rmse += window_mean_rmse
        print("window_avg_error", window_avg_error)

    avg_error = float(hashtag_rmse) / float(window_count)
    print("hashtag avg rmse for file "+ f, avg_error)



