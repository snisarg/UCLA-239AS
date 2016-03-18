
# Q 3 Linear Regression using additional 3 features

import operator
import os
import json
import utility
import numpy
import statsmodels.api as sm
from sklearn import linear_model, cross_validation, neural_network
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

file_list = []

file_list = os.listdir(path)
#file_list = ["subset.txt"]

window_size = 1

for f in file_list:

    print("\nRandom forests Regression on file : ", f)
    print("\nWindow size : ", window_size)
    temp = f
    f = path + f

    training_data = utility.generate_training_data(f, 1, False, 0, 0, True)

    training_data.pop(0)
    #print training_data

    X = utility.get_feature_matrix(training_data, window_size)
    #print X

    rows = X.shape[0]
    #print "shape"
    #print X.shape
    #print "matrix"
    #print X
    # numpy.roll( ) is used for circular shifting of elements
    train_label = X[2:, 0]
    train_features = X[1:- 1, :]

    #model = linear_model.LinearRegression()
    '''
    model = RandomForestRegressor(n_estimators=50,   max_features=7,   max_depth= 8,  n_jobs=8)
    '''
    model = neural_network.MLPRegressor([1,1,1,1,1], 'relu', 'adam', 0.0001, 200, 'constant', 0.001, 0.5, 200, True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08)

    #predicted_tweet_count = cross_validation.cross_val_predict(model, train_features, train_label, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, train_features, train_label,  cv=10, scoring='mean_squared_error')
    #print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("\n", numpy.mean(numpy.sqrt(-scores)))
'''
    # Part B - Generate a Scatter Plot of no of tweets and top 3 features
    p1 = plt.scatter(train_label, train_features[:, 0], s=5, c='#ff0000', linewidths=0)
    p2 = plt.scatter(train_label, train_features[:, 1], s=5, c='#00ff00', linewidths=0)
    p3 = plt.scatter(train_label, train_features[:, 2], s=5, c='#0000ff', linewidths=0)
    plt.xlabel('Tweet Count')
    # plt.ylabel('')
    plt.title('Feature values vs. Predictant for {}'.format(temp))
    #plt.grid(True)
    plt.legend((p1, p2, p3), ('Column name 1', 'Column name 2', 'Column Name 3'))
    plt.show()
'''
    # linear_regression(data)
'''
    regr = sm.OLS(train_label, train_features)
    results = regr.fit()
    print "summary \n"
    print results.summary()
#    predict_label = results.predict(test_features)
'''

    #print ("No of Predicted tweets ", predict_label, "\n")
    #print("\nResidual sum of squares: %.3f"% numpy.mean((predict_label - test_label) ** 2))
    #print("\nMean absolute error |Predicted - Actual|: %.3f"% numpy.mean(numpy.abs(predict_label - test_label)))
'''
    output = "que3-lin-reg-resultc-"
    f = output + temp
    with open(f, 'a') as fw:
        f:q
        w.write(str(results.summary()))
        fw.write("\n\n No of Predicted tweets : ")
        fw.write(str(predict_label))
        fw.write("\n\n Mean absolute error : ")
        fw.write(str(numpy.mean(numpy.abs(list(map(operator.sub, predict_label, test_label))))))
        #fw.write("\n\n Residual sum of squares : ")
'''
