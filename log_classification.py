""" This is a sample code to demonstrate a simple classification using count of words
Steps :  1. Calculate freq of words
         2. Calculate look up for freq of that in train
         3. A simple log regression for classification """






import pandas as pd

import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    m = x.shape[0]

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = (-1 / m) * (np.dot(np.transpose(y), np.log(h)) + np.dot(np.transpose(1 - y), np.log(1 - h)))

        # update the weights theta
        theta = theta - (alpha / m) * (np.dot(np.transpose(x), (h - y)))

    J = float(J)
    return J, theta


def extract_features(tweet, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1


    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0.0)

        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0.0)

    assert (x.shape == (1, 3))
    return x


def train(input_df, text_name, label_name, split=0.80):

    #shuffle

    input_df = input_df.sample(frac=1)

    #split
    no_train = int( len(input_df) * split )

    train_x = input_df[text_name].to_list()[:no_train]
    test_x = input_df[text_name].to_list()[no_train:]

    train_y = np.array(input_df[label_name].to_list()[:no_train])
    test_y = np.array(input_df[label_name].to_list()[no_train:])

    # Freq build
    freqs = build_freqs(train_x, train_y)


    # training

    print("Training Started")
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs)

    # training labels corresponding to X
    Y = train_y

    # Apply gradient descent
    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

    print("Training Completed")

    #testing
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = sum(np.asarray(y_hat) == np.squeeze(test_y)) / len(y_hat)

    print(f"The accruracy of the mode is : {accuracy}")

    return theta , freqs


def predict(text, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''

    # extract the features of the tweet and store it into x
    x = extract_features(text, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))


    return y_pred


if __name__ == '__main__':

    #Some sample data - this will change accrording to data.
    #Note currently this assumes only 2 classes and i.e 1 , 0


    from nltk.corpus import twitter_samples

    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    df_pos = pd.DataFrame({"text" : all_positive_tweets})
    df_pos['label'] = np.ones((len(all_positive_tweets),1))

    df_neg = pd.DataFrame({"text" : all_negative_tweets})
    df_neg['label'] = np.zeros((len(all_negative_tweets),1))

    df = pd.concat([df_pos,df_neg],axis=0)

    theta , freqs  = train(df,text_name='text',label_name='label')

    my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'

    print(process_tweet(my_tweet))
    y_hat = predict(my_tweet, freqs, theta)
    print(y_hat)
    if y_hat > 0.5:
        print('Positive sentiment')
    else:
        print('Negative sentiment')


