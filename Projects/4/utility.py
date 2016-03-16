import os
import json
import datetime


path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

# returns a list of lists of feature data between start and end-time from the input file

# no of tweets
# total retweets
# sum of no of followers of users posting hashtag
# max no of followers of users posting hashtag
# time of the day - 1 of the 24 values with reference to some start  of day

# returns a per hour data as a list of lists of values from entire file which can be used as training data

def generate_training_data(f, hour_window, timeframe, start_time, end_time):

    hours_count = 0
    cur_time = 0
    old_ref_time = 0
    training_data = []
    hour_window_data = []
    max = 0
    print f
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

            # exit condition for 2nd and 3rd Time-frame
            if timeframe is True:
                if (cur_time > end_time) and (end_time != 0):
                    return training_data

            if (timeframe is False) or (timeframe is True and ((start_time < cur_time) or (start_time == 0))):

                # if hour_window is complete, collect stats in list
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
