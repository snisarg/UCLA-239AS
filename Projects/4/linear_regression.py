# Que 2 - Linear Regression
import operator
import os
import json
import datetime
import numpy
import statsmodels.api as sm
import utility
import classify_data
from sklearn import linear_model, cross_validation
import sklearn.pipeline

# Generate training and test data using train_X, train_Y convention for LR
# take no of tweets as train_label & rest of the features as train data or independent vars

path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

file_list = []
file_list = os.listdir(path)
#file_list = ["subset.txt"]

window_size = 1

for f in file_list:

    print("Linear Regression on file : ", f)
    print("window size : ", window_size)
    temp = f
    f = path + f

    training_data = utility.generate_training_data(f, 1, False, 0, 0, False)
    training_data.pop(0)
    #print training_data
    X = utility.get_feature_matrix(training_data, window_size)
    #X = sm.add_constant(X)
    #X = numpy.matrix(training_data)

    rows = X.shape[0]
    #print X
    # numpy.roll( ) is used for circular shifting of elements
    train_label = X[1:, 0]
    train_features = X[: -1, :]
    test_label = X[1:, 0]
    test_features = X[1:rows, :]
    '''
    print "train label"
    print train_label
    print "train features"
    print train_features
    '''

    # linear_regression(data)
    regr = sm.OLS(train_label, train_features.astype(float))
    results = regr.fit()
    print "Summary \n"
    print results.summary()

    '''
    model = linear_model.LinearRegression()
    #model = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), model)
    #predicted_tweet_count = cross_validation.cross_val_predict(model, train_features, train_label, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, train_features, train_label,  cv=10, scoring='mean_absolute_error')
    print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    avg_scores = numpy.average(numpy.abs(scores))
    print("\nMean absolute error : ", avg_scores)

    #predict_label = results.predict(test_features)
    #print ("No of Predicted tweets ", predict_label, "\n")
    #print("\nResidual sum of squares: %.3f"% numpy.mean((predict_label - test_label) ** 2))
    #print("\nMean absolute error |Predicted - Actual|: %.3f"% numpy.mean(numpy.abs(predict_label - test_label)))
    '''
    '''
    output = "linear-reg-result-"
    f = output + temp
    with open(f, 'a') as fw:

    #fw.write(str(results.summary()))

    fw.write("\n\n No of Predicted tweets : ")

    fw.write(str(predict_label))
    fw.write("\n\n Mean absolute error : ")
    fw.write(str(numpy.mean(numpy.abs(list(map(operator.sub, predict_label, test_label))))))
    fw.write("\n\n Residual sum of squares : ")
    #fw.write(str(numpy.mean((predict_label - test_label) ** 2)))
    #fw.write(str(numpy.mean(numpy.sqrt((numpy.square(predict_label[i] - test_features[i] for i in range(len(predict_label))))))))
    '''
