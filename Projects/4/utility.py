import os
import json
import datetime
import numpy
import re
import nltk
from sklearn import feature_extraction
import string

path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

# returns a list of lists of feature data between start and end-time from the input file

# no of tweets
# total retweets
# sum of no of followers of users posting hashtag
# max no of followers of users posting hashtag
# time of the day - 1 of the 24 values with reference to some start  of day

# returns a per hour data as a list of lists of values from entire file which can be used as training data


# converts list of lists to a matrix by aggregating the data of 'window_size'

def get_feature_matrix(training_data, window_size):

    aggr_data = []
    length = len(training_data)
    print training_data[1]
    feature_count = len(training_data[0])

    for i in range(feature_count):
        aggr_data.append(0)


    # Generate sliding window wise aggregate data
    for i in range(length):
        for k in range(feature_count):
            aggr_data[k] = 0
        for j in range(window_size):
            if (i + j) < length:
                aggr_data = numpy.add(aggr_data, training_data[i+j])

        if i == 0:
            X = numpy.matrix(aggr_data)
        else:
            X = numpy.vstack([X, list(aggr_data)])

    return X


def generate_training_data(f, hour_window, timeframe, start_time, end_time, extra_features):

    hours_count = 0
    cur_time = 0
    old_ref_time = 0
    training_data = []
    hour_window_data = []
    unique_users_set = set([])
    unique_tweets_set = set([])
    max = 0
    user_count = 0
    print f

    if extra_features is False:
        no_of_features = 5
    else:
        no_of_features = 9


    for i in range(no_of_features):
       hour_window_data.append(0)

    '''
    hour_window_data :
[0]    tweet_count = 0
[1]    retweet_count = 0
[2]    total_followers = 0
[3]    max_followers = 0
[4]    time_of_day = 0
[5]    friends_count
[6]    no of users posting tweet in current window
[7]    total of favorite count of tweets posted in the current window
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
                    if extra_features is True:
                        hour_window_data[6] = user_count
                    #print ("hour wi data",hour_window_data)
                    training_data.append(list(hour_window_data))
                    max = 0
                    user_count = 0
                    unique_users_set.clear()
                    unique_tweets_set.clear()
                    for i in range(no_of_features):
                        hour_window_data[i] = 0

                if old_ref_time == 0:
                    old_ref_time = cur_time

                # extract hour from UTC time
                # extract curr hour from time and subtract from it 12 am i.e 0, which will be timeofday
                hour_window_data[4] = datetime.datetime.fromtimestamp(cur_time).hour
                # tweet count
                hour_window_data[0] += 1

                # whether a given tweet is retweet or not, to count no of retweets in a window
                if(data["tweet"]["retweet_count"] != 0):
                    hour_window_data[1] += 1

                # add followers count of a user if he is unique in the current window
                if data["tweet"]["user"]["id_str"] not in unique_users_set:
                    unique_users_set.add(data["tweet"]["user"]["id_str"] )
                    foll_count = data["tweet"]["user"]["followers_count"]
                    if foll_count > max:
                        max = foll_count

                    hour_window_data[2] += foll_count

                if extra_features is True:

                  # add followers count of a user if he is unique in the current window

                    if data["tweet"]["user"]["id_str"] not in unique_users_set:
                        unique_users_set.add(data["tweet"]["user"]["id_str"] )
                        hour_window_data[5] += data["tweet"]["user"]["friends_count"]

                        user_count += 1

                    if data["tweet"]["id"] not in unique_tweets_set:
                        unique_tweets_set.add(data["tweet"]["id"])
                        hour_window_data[7] += data["tweet"]["favorite_count"]

    #print training_data
    return training_data


__stemmer = nltk.stem.LancasterStemmer()
__words_only = re.compile("^[A-Za-z]*$")


def punctuation_cleaner(s):
    if s not in string.punctuation:
        return True
    return False


def stop_word_cleaner(s):
    if s not in feature_extraction.text.ENGLISH_STOP_WORDS:
        return True
    return False


def stem_cleaner(s):
    return __stemmer.stem(s)


def clean_word(s):
    result = ""
    if s is not None:
        for w in nltk.tokenize.word_tokenize(s.lower()):
            #print w
            if w is not None and stop_word_cleaner(w) and punctuation_cleaner(w) and regex_filter(w):
                result += " " + stem_cleaner(w)
    #print result
    return result


def regex_filter(s):
    if __words_only.match(s) is not None:
        return True
    return False