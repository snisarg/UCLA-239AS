

import os
import json
import datetime
import csv
import io

#path = "../../Datasets/tweets/tweet_data/"
path = "F:/tweets/"

# returns a list of lists of feature data between start and end-time from the input file

# no of tweets
# total retweets
# sum of no of followers of users posting hashtag
# max no of followers of users posting hashtag
# time of the day - 1 of the 24 values with reference to some start  of day

# returns a per hour data as a list of lists of values from entire file which can be used as training data

#file_list = ["tweets_#gohawks", "tweets_#gopatriots" , "tweets_#nfl", "tweets_#patriots", "tweets_#sb49", "tweets_#superbowl"]
file_list = ["subset.txt"]

for f in file_list:

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

    #fieldnames = ('id', 'text')
    outfile = str(f)
    outfile= path+outfile + ".csv"
   # w = open(outfile, 'w')
    fp = open(outfile, 'wb')
    writer = csv.writer(fp, delimiter=',')
    #writer = csv.DictWriter(f)
    #headers = dict( (n,n) for n in fieldnames )
    #writer.writerow()

    f = path + f

    with io.open(f, 'r', encoding='UTF-8') as f:
        for line in f:
            data = []
            data = json.loads(line)

            #id = data["tweet"]["id"]
            text = data["tweet"]["text"]
            #data.append(id)
            #data.append(text)
            writer.writerow(text.encoding('UTF-8'))

    #print training_data
    fp.close()

