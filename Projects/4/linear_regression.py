import os
import json

# Generate training and test data using train_X, train_Y convention for LR

# no of tweets
# total retweets
# sum of no of followers of users posting hashtag
# max no of followers of users posting hashtag
# time of the day - 1 of the 24 values

path = "../../Datasets/tweets/tweet_data/"

def generate_training_data(file, hour_window):

    i = 0
    hours_count = 0
    cur_time = 0
    old_ref_time = 0
    training_data = [[]]
    hour_window_data = []

    for i in range(5):
       hour_window_data[i] = 0


    '''
    hour_window_data :
[0]    tweet_count = 0
[1]    retweet_count = 0
[2]    total_followers = 0
[3]    max_followers = 0
[4]    time_of_day = 0
    '''

    with open(file, 'r') as f:
        for line in f:

            data.append(json.loads(line))

            cur_time = data[i]["firstpost_date"]

            # when hour_window is complete
            if cur_time > (old_ref_time + (hour_window * (3600))):
                hours_count += 1
                old_ref_time = cur_time
                training_data.append(hour_window_data)
                for i in range(5):
                    hour_window_data[i] = 0

            else:
                hour_window_data[0] += 1
                if(data[i][] )
                    hour_window_data[1] += 1

            i += 1


    return training_data

file_list = []
file_list = os.listdir(path)


for file in file_list:
    file = path + file
    data = generate_training_data(file, 1)

    # linear_regression(data)