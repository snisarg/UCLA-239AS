import os
import json
import datetime
import numpy
from sklearn import linear_model

# Generate training and test data using train_X, train_Y convention for LR
# take no of tweets as train_label & rest of the features as train data or independent vars

path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

# no of tweets
# total retweets
# sum of no of followers of users posting hashtag
# max no of followers of users posting hashtag
# time of the day - 1 of the 24 values with reference to some start  of day

# returns a list of lists of values from entire file which are used as training data
# change the window logic

def generate_training_data(f, hour_window):

    hours_count = 0
    cur_time = 0
    old_ref_time = 0
    training_data = []
    hour_window_data = []
    max = 0
    print  f
    for i in range(5):
       hour_window_data.append(0)

    '''
    hour_window_data :
[0]    tweet_count = 0
[1]    retweet_count = 0
[2]    total_followers = 0
[3]    max_followers = 0
[4]    time_of_day = 0

Start reference for time of the day 12 am
    '''

    with open(f, 'r') as f:
        for line in f:

            data = json.loads(line)

            cur_time = data["firstpost_date"]
            #print cur_time
            # when hour_window is complete
            if (old_ref_time != 0) and (cur_time > (old_ref_time + (hour_window * (3600)))):
                hours_count += 1
                old_ref_time = cur_time
                hour_window_data[3] = max
                #print ("hour wi data",hour_window_data)
                training_data.append(list(hour_window_data))
                max = 0
                for i in range(5):
                    hour_window_data[i] = 0
            if old_ref_time == 0:
                old_ref_time = cur_time

            # extract hour from UTC time
            # extract curr hour from time and subtract from it 12 am i.e 0, which will be timeofday
                #hour_window_data[4]
            hour_window_data[4] = datetime.datetime.fromtimestamp(cur_time).hour
            # tweet count
            hour_window_data[0] += 1

            # whether a given tweet is retweet or not, to count no of retweets in a window
            if(data["tweet"]["retweet_count"] != 0):
                hour_window_data[1] += 1

            foll_count = data["tweet"]["user"]["followers_count"]
            if foll_count > max:
                max = foll_count

            hour_window_data[2] += foll_count
    #print training_data
    return training_data

file_list = []

file_list = os.listdir(path)
#file_list = ["subset.txt"]

window_size = 5

for f in file_list:


    print("Linear Regression on file", f)
    print("window size", window_size)

    f = path + f

    training_data = generate_training_data(f, 1)
    training_data.pop(0)
#    print training_data
    X = numpy.matrix(training_data)
    rows = X.shape[0]
#    print("rows", rows)
    #cols = X.shape[1]
    #print("No of observations :", rows, "cols :", cols)
#    print X

    for i in range(rows - 1):
        window_end = i + window_size - 1
        if window_end >= rows:
            break
        train_label = X[i: window_end, 0]
        train_features = X[i: window_end, [1, 4]]
        test_label = X[window_end, 0]
        test_features = X[ window_end , [1,4]]

        # linear_regression(data)
        model = linear_model.LinearRegression()
        model.fit(train_features, train_label)

        print("Co-efficients : ", model.coef_)
        print("Residual sum of squares: %.2f"% numpy.mean((model.predict(test_features) - test_label) ** 2))
        #print('Variance score: %.2f'% model.score(test_features, test_label))

    # train & fit model
    # compare with test data i.e no of tweets in next hour