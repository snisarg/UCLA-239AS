# Que 2 - Linear Regression

import os
import json
import datetime
import numpy
import statsmodels.api as sm
import utility

# Generate training and test data using train_X, train_Y convention for LR
# take no of tweets as train_label & rest of the features as train data or independent vars

#path = "../../Datasets/tweets/tweet_data/"
path = "F:/tweets/"

file_list = []
#file_list = os.listdir(path)
file_list = ["subset.txt"]

window_size = 1

for f in file_list:

    print("Linear Regression on file : ", f)
    print("window size : ", window_size)
    temp = f
    f = path + f

    training_data = utility.generate_training_data(f, 1, False, 0, 0, False)
    training_data.pop(0)
    print training_data
    X = utility.get_feature_matrix(training_data, window_size)
    #X = sm.add_constant(X)
    #X = numpy.matrix(training_data)
    rows = X.shape[0]
    print "shape"
    print X.shape
    print "matrix"
    print X
    # numpy.roll( ) is used for circular shifting of elements

    train_label = X[:rows - 2, 0]
    train_features = X[: rows - 2, 1:5]
    test_label = X[ rows - 1, 0]
    test_features = X[rows-1, 1:5]

    print "test label"
    print test_label
    print "test features"
    print test_features
    # linear_regression(data)
    regr = sm.OLS(train_label, train_features.astype(float))
    results = regr.fit()
    print "summary \n"
    print results.summary()
    predict_label = results.predict(test_features)
    #print ("No of Predicted tweets ", predict_label, "\n")
    #print("\nResidual sum of squares: %.3f"% numpy.mean((predict_label - test_label) ** 2))
    #print("\nMean absolute error |Predicted - Actual|: %.3f"% numpy.mean(numpy.abs(predict_label - test_label)))

    output = "linear-reg-result-"
    f = output + temp
    with open(f, 'a') as fw:
        fw.write(str(results.summary()))
        fw.write("\n No of Predicted tweets : ")
        fw.write(str(predict_label))
        fw.write("\n Mean absolute error : ")
        fw.write(str(numpy.mean(numpy.abs(predict_label - test_label))))
        fw.write("\n Residual sum of squares : ")
        fw.write(str(numpy.mean((predict_label - test_label) ** 2)))
