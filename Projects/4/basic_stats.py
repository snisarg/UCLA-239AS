
# scp test_data.tar.gz nisarg@192.168.0.4:~/z/EE239AS/Datasets/tweets

import json
import os
# Calculate average number of tweets per hour

path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

# Find average number of retweets
retweet_count = 0
file_list = []
file_list = os.listdir(path)

print file_list
for file in file_list:

    #if file == "subset.txt":

        #data = []
        file = path + file
        hours_count = -1
        retweet_count = 0
        followers_count = 0
        unique_users = []
        cur_time = 0
        count = 0
        old_ref_time = 0
        #with open("F:/tweets/tweet_data/subset.txt",'r') as f:
        with open(file, 'r') as f:
            for line in f:

                data= json.loads(line)

                # Count avg tweets per hour
                cur_time = data["firstpost_date"]

                if cur_time > (old_ref_time + 3600):
                    hours_count += 1
                    old_ref_time = cur_time

                # count followers of original authors of tweets
                author = data["original_author"]["url"]
                if author not in unique_users:
                    unique_users.append(author)
                    followers_count += data["original_author"]["followers"]

                # count no of retweets
                retweet_count += data["tweet"]["retweet_count"]
                count += 1
            #print i
        total_tweets_count = count
        avg_tweets_per_hour = float(float(total_tweets_count) / float(hours_count))
        avg_retweets = float(float(retweet_count) / float(total_tweets_count))
        avg_followers_count = float(float(followers_count) / float(total_tweets_count))
        print "average tweets per hour for "+ file
        print avg_tweets_per_hour
        print "average retweets for "+ file
        print avg_retweets
        print "average followers count for "+ file
        print avg_followers_count

    #break

'''
Output :

average tweets per hour for ../../Datasets/tweets/tweet_data/tweets_#gohawks.txt
301.5
average retweets for ../../Datasets/tweets/tweet_data/tweets_#gohawks.txt
0.20916252073
average followers count for ../../Datasets/tweets/tweet_data/tweets_#gohawks.txt
720.952109113
average tweets per hour for ../../Datasets/tweets/tweet_data/tweets_#gopatriots.txt
68.3125
average retweets for ../../Datasets/tweets/tweet_data/tweets_#gopatriots.txt
0.0268374504422
average followers count for ../../Datasets/tweets/tweet_data/tweets_#gopatriots.txt
1068.15953797
average tweets per hour for ../../Datasets/tweets/tweet_data/tweets_#nfl.txt
424.629508197
average retweets for ../../Datasets/tweets/tweet_data/tweets_#nfl.txt
0.0509373648774
average followers count for ../../Datasets/tweets/tweet_data/tweets_#nfl.txt
1304.77502857



'''