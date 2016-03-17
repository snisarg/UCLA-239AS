import json
from pandas.io.json import json_normalize
import codecs
import pandas
import utility

# result_file = codecs.open('../../Datasets/tweets/generated/Tweets_only.csv', 'a')
tweets = []
with codecs.open('../../Datasets/tweets/tweet_data/tweets_#gopatriots.txt') as data_file:
    for line in data_file:
        tweet_obj = json_normalize(json.loads(line)['tweet'])
        # data.append(pandas.Series([tweet_obj['text'][0]]))
        tweets.append(utility.clean_word(tweet_obj['text'][0]))
        #result_file.write(unicode(str(tweet_obj['text'][0])))
        #result_file.write('\n')
        # print tweet_obj['text'][0]

# print data.head()
# result_file.close()
data = pandas.Series(tweets)
data.to_csv('../../Datasets/tweets/generated/Tweets_only.csv', encoding='utf-8')

