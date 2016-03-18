
# Q 3 Linear Regression using additional 3 features

import operator
import os
import json
import utility
import numpy
import statsmodels.api as sm

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

    training_data = utility.generate_training_data(f, 1, False, 0, 0, True)

    training_data.pop(0)
    #print training_data

    X = utility.get_feature_matrix(training_data, window_size)
    print X
    #X = sm.add_constant(X)
    #X = numpy.matrix(training_data)

    rows = X.shape[0]
    #print "shape"
    #print X.shape
    #print "matrix"
    #print X
    # numpy.roll( ) is used for circular shifting of elements
    train_label = X[2:, 8]
    train_features = X[1:- 1, :]
    #test_label = X[1:, 0]
    #test_features = X[1:, :]
    '''
    print "train label"
    print train_label
    print "train features"
    print train_features
    '''
    # linear_regression(data)
    regr = sm.OLS(train_label, train_features.astype(float))
    results = regr.fit()
    print "summary \n"
    print results.summary()
    #predict_label = results.predict(test_features)
    #print ("No of Predicted tweets ", predict_label, "\n")
    #print("\nResidual sum of squares: %.3f"% numpy.mean((predict_label - test_label) ** 2))
    #print("\nMean absolute error |Predicted - Actual|: %.3f"% numpy.mean(numpy.abs(predict_label - test_label)))
'''
    output = "que3-lin-reg-resultc-"
    f = output + temp
    with open(f, 'a') as fw:
        fw.write(str(results.summary()))
        fw.write("\n\n No of Predicted tweets : ")
        fw.write(str(predict_label))
        fw.write("\n\n Mean absolute error : ")
        fw.write(str(numpy.mean(numpy.abs(list(map(operator.sub, predict_label, test_label))))))
        #fw.write("\n\n Residual sum of squares : ")
'''

# Part B - Generate a Scatter Plot of no of tweets and top 3 features

