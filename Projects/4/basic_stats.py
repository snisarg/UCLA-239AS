import json
import os
import io
# Calculate average number of tweets per hour

path = "../../Datasets/tweets/tweet_data/"
#path = "F:/tweets/"

def hashtag_stats(f):
    retweet_count = 0

    #if file == "subset.txt":

    #data = []
    f = path + f
    hours_count = -1
    retweet_count = 0
    followers_count = 0
    unique_users = set([])
    cur_time = 0
    count = 0
    old_ref_time = 0
    #with open("F:/tweets/tweet_data/subset.txt",'r') as f:
    for line in io.open(f, encoding="utf8"):

        data= json.loads(line)
        # Count avg tweets per hour
        cur_time = data["firstpost_date"]

        if cur_time > (old_ref_time + 3600):
            hours_count += 1
            old_ref_time = cur_time

        # count followers of original authors of tweets
        author = data["author"]["url"]
        if author not in unique_users:
            unique_users.add(author)
            followers_count += data["author"]["followers"]

        # count no of retweets
        retweet_count += data["metrics"]["citations"]["total"]
        count += 1
        #print i
    total_tweets_count = count
    avg_tweets_per_hour = float(float(total_tweets_count) / float(hours_count))
    avg_retweets = float(float(retweet_count) / float(total_tweets_count))
    avg_followers_count = float(float(followers_count) / float(total_tweets_count))
    print "\naverage tweets per hour for "+ f
    print avg_tweets_per_hour
    print "\naverage retweets for "+ f
    print avg_retweets
    print "\naverage followers count for "+ f
    print avg_followers_count

    #break


file_list = []
file_list = os.listdir(path)

print file_list
for f in file_list:
    hashtag_stats(f)

'''






Old Output :

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