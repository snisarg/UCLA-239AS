import os
import json

# Generate training and test data using train_X, train_Y convention for LR
# take no of tweets as train_label & rest of the features as train data or independent vars



# no of tweets
# total retweets
# sum of no of followers of users posting hashtag
# max no of followers of users posting hashtag
# time of the day - 1 of the 24 values with reference to some start  of day


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

Start reference for time of the day 12 am :
    '''

    with open(file, 'r') as f:
        for line in f:

            data.append(json.loads(line))

            cur_time = data[i]["firstpost_date"]

            # when hour_window is complete
            if cur_time > (old_ref_time + (hour_window * (3600))):
                hours_count += 1
                old_ref_time = cur_time
                hour_window_data[3] = max
                training_data.append(hour_window_data)
                max = 0
                for i in range(5):
                    hour_window_data[i] = 0

            else:
                hour_window_data[0] += 1
                if(data[i]["tweet"]["retweet_count"] != 0)
                    hour_window_data[1] += 1

                foll_count = data[i]["tweet"]["user"]["followers_count"]
                if foll_count > max:
                    max = foll_count

                hour_window_data[2] += foll_count

                # extract curr hour from time and subtract from it 12 am i.e 0, which will be timeofday
                #hour_window_data[4]

            i += 1


    return training_data

file_list = []
file_list = os.listdir(path)


for file in file_list:
    file = path + file
    data = generate_training_data(file, 1)

    # linear_regression(data)

    # train & fit model
    # compare with test data i.e no of tweets in next hour