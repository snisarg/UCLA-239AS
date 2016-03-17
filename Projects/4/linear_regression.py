

# Que 2 - Linear Regression

import os
import json
import datetime
import numpy
import statsmodels.api as sm
import utility

# Generate training and test data using train_X, train_Y convention for LR
# take no of tweets as train_label & rest of the features as train data or independent vars

path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

file_list = []
file_list = os.listdir(path)
#file_list = ["subset.txt"]

window_size = 5

for f in file_list:

    print("Linear Regression on file : ", f)
    print("window size : ", window_size)
    temp = f
    f = path + f

    training_data = utility.generate_training_data(f, 1, False, 0, 0, False)
    training_data.pop(0)
    print training_data
    X = utility.get_feature_matrix(training_data, window_size)

    #X = numpy.matrix(training_data)
    rows = X.shape[0]

    # numpy.roll( ) is used for circular shifting of elements

    train_label = X[:rows - 2 , 0]
    train_features = X[: rows - 2, [1, 4]]
    test_label = X[ rows - 1, 0]
    test_features = X[rows-1, [1,4]]

        # linear_regression(data)
    regr = sm.OLS(train_label, train_features)
    results = regr.fit()
    print "summary \n"
    print results.summary()
    predict_label = results.predict(test_features)
    print ("no of tweets ", predict_label, "\n")
    print("\nResidual sum of squares: %.2f"% numpy.mean((predict_label - test_label) ** 2))

    output = "linear-reg-result-"
    f = output + temp
    with open(f, 'a') as fw:
        fw.write(str(results.summary()))
        fw.write(" No of tweets: ")
        fw.write(str(predict_label))
        fw.write(" Residual sum of squares : ")
        fw.write(str(numpy.mean((predict_label - test_label) ** 2)))

