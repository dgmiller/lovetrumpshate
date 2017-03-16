import re
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer as Vec
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import string
import nltk
import datetime
import time
from matplotlib import pyplot as plt

# filepaths of corresponding OS
labloc = '/run/media/derekgm@byu.local/FAMHIST/Data/final_project/'
mintloc = '/media/derek/FAMHIST/Data/final_project/'
# filenames
trumplab = labloc + 'trump.txt'
clintonlab = labloc + 'cleantrump.txt'
trumpmint = mintloc + 'trump.txt'
clintonmint = mintloc + 'cleantrump.csv'

def get_file():
    print("""\n\tOptions\n
            1: trump from lab computer\n
            2: trump from linux mint\n
            3: clean trump from lab computer\n
            4: clean trump from linux mint\n\n""")
    name = raw_input("Enter number >> ")
    if name == "1":
        return trumplab
    elif name == "2":
        return trumpmint
    elif name == "3":
        return clintonlab
    elif name == "4":
        return clintonmint
    else:
        print "invalid input"

class TwitterCorpus(object):
    
    def __init__(self,filename,n=None,m=None):
        """
        ATTRIBUTES:
            data (string) the tweet data from a txt file
            tweets (list) of user tweet content
            user_stats (list) of number of followers, friends, and user tweets to date
            timestamps (list) of floats indicating UTC timestamp
            time (list) of timestamps converted to datetime objects
            n_mentions (list) of the number of mentions in a tweet
            n_hashtags (list) of the number of hashtags in a tweet
            n_weblinks (list) of the number of external links in a tweet
            retweets (list) of booleans indicating whether the tweet was a retweet
        """
        print("Loading file...\n")
        start = time.time()
        self.data = open(filename,'r').readlines()[n:m]
        self.tweets = []
        self.user_stats = []
        self.timestamps = []
        self.time = []
        err = 0
        for i,line in enumerate(self.data):
            line = line.split('\t')
            # get everything except for the tweet
            try:
                # number of followers, statuses, and friends
                self.user_stats.append([float(j) for j in line[1:-1]])
                # time that the tweet was sent
                self.timestamps.append(int(line[0]))
                # content of the tweet
                self.tweets.append(line[-1])
            except:
                print i,line
                err += 1
        print "Errors: " + str(err)
        # convert to numpy array
        # self.timestamps = np.array(self.timestamps)
        self.user_stats = np.array(self.user_stats)
        self.n_mentions = []
        self.n_hashtags = []
        self.n_weblinks = []
        self.retweets = []
        end = time.time()
        print("Time: %s" % (end-start))
        
    def clean_text(self,remove_retweets=True):
        """
        Cleans the text and extracts information from the tweet
        """
        print("Cleaning text...")
        start = time.time()
        tweetwords = []
        u_h = []
        u_m = []
        for s in self.tweets:
            m_str = ""
            h_str = ""
            s = s.replace('"""','')
            mentions = re.findall(r'@\w*',s)
            hashtags = re.findall(r'#\w*',s)
            weblinks = re.findall(r'http\S*',s)
            retweets = re.findall('^RT ',s)
            numbers = re.findall(r'[0-9]+',s)
            self.n_mentions.append(len(mentions))
            self.n_hashtags.append(len(hashtags))
            self.n_weblinks.append(len(weblinks))
            self.retweets.append(len(retweets))
            for m in mentions:
                u_m.append(m)
            for h in hashtags:
                u_h.append(h)
            tweetwords.append(s)
        self.mentions = u_m
        self.hashtags = u_h
        self.u_mentions = np.unique(u_m)
        self.u_hashtags = np.unique(u_h)
        self.tweets = tweetwords
        end = time.time()
        print("Time: %s" % (end-start))
        
    def remove_keywords(self, keywords):
        """
        Remove the specified keywords from the list. Updates self.tweets
        INPUT
            keywords (list) of keywords to remove from tweets
        """
        new_tweets = []
        for t in self.tweets:
            for k in keywords:
                t = t.lower().replace(k.lower(),"")
            new_tweets.append(t)
        self.tweets = new_tweets
    
    def convert_time(self):
        """
        converts timestamp to datetime object, stored as self.time
        """
        print("Converting time to datetime object...")
        start = time.time()
        self.time = pd.to_datetime(self.timestamps,unit='ms') - pd.DateOffset(hours=7)
        end = time.time()
        print("Time: %s" % (end-start))
        
    def get_sentiment(self):
        """
        How do these tweets make you feel? Sentiment scores from nltk.
        RETURNS
            neg,pos,comp (arrays) negative, positive, and compound sentiment scores
        """
        neg = []
        neu = []
        pos = []
        comp = []
        S = SentimentIntensityAnalyzer()
        for tweet in self.tweets:
            S_ = S.polarity_scores(tweet)
            neg.append(S_['neg'])
            neu.append(S_['neu'])
            pos.append(S_['pos'])
            comp.append(S_['compound'])
        return neg,neu,pos,comp

    def make_df(self,time_index=True):
        """
        Creates a dataframe of the twitter data
        INPUT
            time_index (bool) whether to set the dataframe index as the time variable, default=False
        RETURNS
            df (pandas DataFrame) of twitter data
        """
        print("Creating DataFrame...")
        start = time.time()
        if time_index:
            df = pd.DataFrame(index=self.time)
        else:
            df = pd.DataFrame()
            df['time'] = self.time
        df['ts'] = self.timestamps
        df['usr_fol'] = self.user_stats[:,0]
        df['usr_n_stat'] = self.user_stats[:,1]
        df['usr_fri'] = self.user_stats[:,2]
        df['n_weblinks'] = self.n_weblinks
        df['n_mentions'] = self.n_mentions
        df['n_hashtags'] = self.n_hashtags
        df['RT'] = self.retweets
        neg,neu,pos,comp = self.get_sentiment()
        df['neg'] = neg
        df['neu'] = neu
        df['pos'] = pos
        df['comp'] = comp
        df['text'] = self.tweets
        end = time.time()
        print("Time: %s" % (end-start))
        return df

def make_train_test(n=None,m=None):
    """
    Returns two dataframes with prepared data for machine learning.
    INPUT
        n (int) index to start at, default=None
        m (int) index to end on, default=None
    OUTPUT
        df (dataframe) full dataframe with cleaned text
        T (dataframe) subset of df where retweets are removed pos and neg columns combined
    """
    filename = get_file()
    c = TwitterCorpus(filename,n,m)
    c.clean_text()
    c.convert_time()
    df = c.make_df()
    T = df[df['RT']==0]
    T = T[((T['pos']==0) & (T['neg']>0)) | ((T['neg']==0) & (T['pos']>0))]
    T['pos-neg'] = T['pos'] - T['neg']
    T.drop(['neg','pos','RT'],axis=1,inplace=True)
    return c,df,T

