import os
import json
import datetime
# Generate training and test data using train_X, train_Y convention for LR
# take no of tweets as train_label & rest of the features as train data or independent vars

path = "../../Datasets/tweets/tweet_data/"



# no of tweets
# total retweets
# sum of no of followers of users posting hashtag
# max no of followers of users posting hashtag
# time of the day - 1 of the 24 values with reference to some start  of day

# returns a list of lists of values from entire file which are used as training data
# change the window logic

def generate_training_data(file, hour_window):


    hours_count = 0
    cur_time = 0
    old_ref_time = 0
    training_data = [[]]
    hour_window_data = []
    tweet_counts= []
    retweet_counts = []
    total_followers = []
    max_followers = []
    times_of_day = []

    for i in range(5):
       hour_window_data[i] = 0

    '''
    hour_window_data :
[0]    tweet_count = 0
[1]    retweet_count = 0
[2]    total_followers = 0
[3]    max_followers = 0
[4]    time_of_day = 0

Start reference for time of the day 12 am
    '''

    with open(file, 'r') as f:
        for line in f:

            data = json.loads(line)

            cur_time = data["firstpost_date"]

            # when hour_window is complete
            if cur_time > (old_ref_time + (hour_window * (3600))):
                hours_count += 1
                old_ref_time = cur_time
                hour_window_data[3] = max
                tweet_counts.append( hour_window_data[0])
                retweet_counts.append( hour_window_data[1])
                total_followers.append(hour_window_data[2])
                max_followers.append( hour_window_data[3])
                times_of_day.append(hour_window_data[4])

                max = 0
                for i in range(5):
                    hour_window_data[i] = 0

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

    return training_data

file_list = []
file_list = os.listdir(path)

tweet_counts= []
retweet_counts = []
total_followers = []
max_followers = []
times_of_day = []

train_labels = []
train_features = [[]]

for file in file_list:
    file = path + file

    tweet_counts, retweet_counts, total_followers, max_followers, times_of_day = generate_training_data(file, 1)
    # linear_regression(data)

    # train & fit model
    # compare with test data i.e no of tweets in next hour