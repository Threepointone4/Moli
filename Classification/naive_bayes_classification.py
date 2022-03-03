"""
    This is a sample code to demonstrate a naive bayes classification

    Steps :  1. Calculate lambda for each word and store it in dict from train set
             2. Calculate look up for lambda of that in train
             3. Calculate log likehood for test

"""

# CHECK __main__ ( in the end to see how to use this )

from utils import *


def count_tweets(result, tweets, ys):
    """
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    """

    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)

            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1

    return result


def train_naive_bayes(freqs, train_x, train_y):
    """
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    """
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]

    # Calculate D, the number of documents
    D = len(train_y)

    # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
    D_pos = (len(list(filter(lambda x: x > 0, train_y))))

    # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
    D_neg = (len(list(filter(lambda x: x <= 0, train_y))))

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood


def naive_bayes_predict(tweet, logprior, loglikelihood):
    """
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    """
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p


if __name__ == '__main__':

    # Some sample data - this will change accrording to data.
    # Note currently this assumes only 2 classes and i.e 1 , 0

    from nltk.corpus import twitter_samples

    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    # split the data into two pieces, one for training and one for testing (validation set)
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # avoid assumptions about the length of all_positive_tweets
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
    test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

    freqs = count_tweets({}, train_x, train_y)
    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
    my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'

    print(process_tweet(my_tweet))
    y_hat = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    print(y_hat)
    if y_hat > 0:
        print('Positive sentiment')
    else:
        print('Negative sentiment')
