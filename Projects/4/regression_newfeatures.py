
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

file_names = []
acceleration_tweets = []
impression_tweets = []
users_tweets = []
label_tweets = []

window_size = 1

for f in file_list:

    print("\nLinear Regression with imp features, on file : ", f)
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
    train_features = X[1:- 1, [6,7,8,9,10,11,12,13]]

    model = linear_model.LinearRegression()
    '''
    model = RandomForestRegressor(n_estimators=50,   max_features=7,   max_depth= 8,  n_jobs=8)
    '''
    #model = neural_network.MLPRegressor([20,10], 'relu', 'adam', 0.0001, 200, 'constant', 0.001, 0.5, 200, True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08)
    '''
    regr = sm.OLS(train_label, train_features)
    results = regr.fit()
    print "summary \n"
    print results.summary()
    '''

    #predicted_tweet_count = cross_validation.cross_val_predict(model, train_features, train_label, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, train_features, train_label,  cv=10, scoring='mean_squared_error')
    #print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("\n", numpy.mean(numpy.sqrt(-scores)))

    # Collecting stats for Part B
    file_names.append(temp)
    label_tweets.append(train_label)
    acceleration_tweets.append(train_features[:, 5])
    impression_tweets.append(train_features[:, 6])
    users_tweets.append(train_features[:, 1])


    # linear_regression(data)

#    predict_label = results.predict(test_features)


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


# Part B - Generate a Scatter Plot of no of tweets and top 3 features
# p1 = plt.scatter(train_label, train_features[:, 5], s=5, c='#ff0000', linewidths=0)
# p2 = plt.scatter(train_label, train_features[:, 6], s=5, c='#001e00', linewidths=0)
# p3 = plt.scatter(train_label, train_features[:, 1], s=5, c='#0000ff', linewidths=0)

colour = ['#ff0000', '#996633', '#009933', '#000066', '#0099ff', '#cc00cc', '#0000ff']

plt.xlabel('Tweet Count')

# For Acceleration of Tweets
plt.ylabel('Acceleration')
plt.title('Average Acceleration of Tweets')
legend_list = []
for i in range(len(acceleration_tweets)):
    legend_list.append(plt.scatter(label_tweets[i], acceleration_tweets[i], c=colour[i], s=5, linewidths=0))
plt.legend(legend_list, file_names)

#plt.grid(True)
# plt.legend((p1, p2, p3), ('Average Acceleration of Tweets', 'Sum of Impressions of Tweets', 'No of users posting Tweets'))
plt.show()


plt.xlabel('Tweet Count')

# For Impressions of Tweets
plt.ylabel('Impressions')
plt.title('Sum of Impressions of Tweets')
legend_list = []
for i in range(len(acceleration_tweets)):
    legend_list.append(plt.scatter(label_tweets[i], impression_tweets[i], c=colour[i], s=5, linewidths=0))
plt.legend(legend_list, file_names)
plt.show()

# For Users of Tweets
plt.xlabel('Tweet Count')
plt.ylabel('Users')
plt.title('Number of users posting Tweets')
legend_list = []
for i in range(len(acceleration_tweets)):
    legend_list.append(plt.scatter(label_tweets[i], users_tweets[i], c=colour[i], s=5, linewidths=0))
plt.legend(legend_list, file_names)
plt.show()