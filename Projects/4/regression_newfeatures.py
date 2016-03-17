
# Q 3 Linear Regression using additional 3 features

import os
import json
import utility
import numpy
import statsmodels.api as sm


path = "../../Datasets/tweets/tweet_data/"


file_list = []

file_list = os.listdir(path)
#file_list = ["subset.txt"]

window_size = 11

for f in file_list:


    print("Linear Regression on file", f)
    print("window size", window_size)
    temp = f
    f = path + f

    training_data = utility.generate_training_data(f, 1, False, 0, 0, True)
    training_data.pop(0)

    X = numpy.matrix(training_data)
    rows = X.shape[0]

    # numpy.roll( ) is used for circular shifting of elements


    for i in range(rows - 1):
        window_end = i + window_size - 1
        if window_end >= rows:
            break
        train_label = X[i: window_end, 0]
        train_features = X[i: window_end, 1 : 8]
        test_label = X[window_end, 0]
        test_features = X[ window_end , 1:8]

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


# Part B - Generate a Scatter Plot of no of tweets and top 3 features

