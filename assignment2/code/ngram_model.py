#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0
    ### YOUR CODE HERE
    
    for doc in dataset:
        # Generate lists of unigrams, bigrams, trigrams
        unigrams = doc[1:]; # we don't count the first 'start' token
        bigrams = zip(doc, doc[1:])
        trigrams = zip(doc, doc[1:], doc[2:])
        token_count += len(doc[2:]) # we count all words except for the to start tokens
        
        # Incement relevant dictionary entries
        for uni in unigrams:
            unigram_counts[uni] = unigram_counts.get(uni, 0) + 1
        for bi in bigrams:
            bigram_counts[bi] = bigram_counts.get(bi, 0) + 1
        for tri in trigrams:
            trigram_counts[tri] = trigram_counts.get(tri, 0) + 1
        
            
    ### END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    ### YOUR CODE HERE
    
    # lambda1 is for trigrams, lambda2 is for bigrams, lambda3 is for unigrams
    lambda3 = 1 - lambda1 - lambda2
    log_sum = 0
    M = 0 # M goes over all trigrams
   
    for doc in eval_dataset:    
        for trigram in zip(doc, doc[1:], doc[2:]):
            word1 = trigram[0] # w_(i-2)
            word2 = trigram[1] # w_(i-1)
            word3 = trigram[2] # w_i
            
            if (bigram_counts.get((word1,word2),0) == 0):
                trigram_prob = 0.0
            else:
                trigram_prob = trigram_counts.get((word1,word2,word3),0) / float(bigram_counts.get((word1,word2),0))
                
            if (unigram_counts.get((word2),0) == 0):
                bigram_prob = 0.0
            else:
                bigram_prob = bigram_counts.get((word2,word3),0) / float(unigram_counts.get((word2),0))
            
            unigram_prob = unigram_counts.get((word3),0) / float(train_token_count)

            M += 1
            log_sum += np.log2(lambda1 * trigram_prob + \
                               lambda2 * bigram_prob + \
                               lambda3 * unigram_prob)
    
    perplexity = 2 ** (-((1/float(M))*(log_sum)))
    
    ### END YOUR CODE
    return perplexity

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    
    print "#Performing grid search on lambda values..."
    delta = 0.05
    lambda1_vals = np.arange(0,1,delta)
    lambda2_vals = np.arange(0,1,delta)
    
    min_perplexity_val = float('inf')
    min_perplexity_lambda1 = -1
    min_perplexity_lambda2 = -2
    
    print
    print '\t\t',
    print '\t'.join([('l1=%.2f' % l1val) for l1val in lambda1_vals])
    for l2val in lambda2_vals:
        print ('l2=%.2f' % l2val),
        for l1val in lambda1_vals:
            print '\t',
            if l1val + l2val >= 1:
                # lambda3 <= 0, invalid (we must give some value to unigrams,
                # because some bigrams / trigrams never appear in training
                print '-',
            else:
                perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts,\
                                             unigram_counts, token_count, l1val, l2val)
                if perplexity < min_perplexity_val:
                    min_perplexity_val = perplexity
                    min_perplexity_lambda1 = l1val
                    min_perplexity_lambda2 = l2val
                print ('%.2f' % perplexity),
        print
    
    print
    print '#minimal perplexity %.2f obtained for lambda1=%.2f, lambda2=%.2f, lambda3=%.2f' % \
                (min_perplexity_val,min_perplexity_lambda1,min_perplexity_lambda2,\
                 1-min_perplexity_lambda1-min_perplexity_lambda2)
    print '#perplexity for training only unigrams (lambda1=0, lambda2=0, lambda3=1) is: %.2f' % \
                evaluate_ngrams(S_dev, trigram_counts, bigram_counts,\
                                             unigram_counts, token_count, 0, 0)
    print '#perplexity for training only bigrams is undefined - there are some bigrams never seen during training'
    
    ### END YOUR CODE

if __name__ == "__main__":
    test_ngram()
