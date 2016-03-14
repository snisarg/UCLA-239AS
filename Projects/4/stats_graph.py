
# scp test_data.tar.gz nisarg@192.168.0.4:~/z/EE239AS/Datasets/tweets

import json
import os
import matplotlib.pyplot as plt

def plot_graph(hashtag, data):

    plt.figure(figsize=(25, 10))
    plt.bar(range(len(data)), data, width=2, color='r')
    plt.title(hashtag)
    plt.xlabel('Hours')
    plt.ylabel('Tweets')
    plt.show()


path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

# Find average number of retweets
retweet_count = 0
file_list = []
file_list = os.listdir(path)

print file_list
gohawks_list = []
nfl_list = []

for file in file_list:

    #if file == "subset.txt":
    if file == "tweets_#gohawks.txt" or file == "tweets_#nfl.txt":
        print "in condition"
        data = []
        file = path + file
        hours_count = -1
        cur_time = 0
        gohawks = 0
        nfl = 0

        i = 0
        old_ref_time = 0
        #with open("F:/tweets/tweet_data/subset.txt",'r') as f:
        with open(file, 'r') as f:
            for line in f:

                data.append(json.loads(line))

                # Count avg tweets per hour
                cur_time = data[i]["firstpost_date"]

                if cur_time > (old_ref_time + 3600):
                    hours_count += 1
                    old_ref_time = cur_time

                    if file == "tweets_#gohawks.txt" :
                        gohawks_list.append(gohawks)
                        go_hawks = 0

                    if file == "tweets_#nfl.txt":
                        nfl_list.append(nfl)
                        nfl = 0
                else :
                    if file ==  "tweets_#gohawks.txt" :
                       go_hawks += 1

                    if file == "tweets_#nfl.txt":
                        nfl += 1

                # count followers of original authors of tweets
                i += 1
                #print i

print "gohawks list"
print gohawks_list
print "nfl list"
print nfl_list

plot_graph('gohawks', gohawks_list)
plot_graph('nfl',nfl_list)

    #break