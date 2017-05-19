import numpy as np
from data import *

TRAIN       = False
START_TOKEN = '*'
STOP_TOKEN  = '$$$'
START_TAG   = '<start>'
STOP_TAG    = '<stop>'
START       = (START_TOKEN, START_TAG)
STOP        = (STOP_TOKEN, STOP_TAG)

q_cache             = None
tags_for_word_cache = None

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Rerutns: the q-counts and e-counts of the sentences' tags
    """
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    
    for sent in sents:
        total_tokens += len(sent)
        sent = [START, START] + sent + [STOP]
        uni  = sent
        bi   = zip(sent, sent[1:])
        tri  = zip(sent, sent[1:], sent[2:])
        
        for w_t in uni:
            word, tag = w_t
            q_uni_counts[tag] = q_uni_counts.get(tag, 0) + 1
        
        for w_t1, w_t2 in bi:
            word1, tag1 = w_t1
            word2, tag2 = w_t2
            q_bi_counts[(tag1, tag2)] = q_bi_counts.get((tag1, tag2), 0) + 1
        
        for w_t1, w_t2, w_t3 in tri:
            word1, tag1 = w_t1
            word2, tag2 = w_t2
            word3, tag3 = w_t3
            q_tri_counts[(tag1, tag2, tag3)] = q_tri_counts.get((tag1, tag2, tag3), 0) + 1
            
        for word, tag in sent[2:-1]:
            e_word_tag_counts[(word, tag)]  = e_word_tag_counts.get((word, tag), 0) + 1
            e_tag_counts[tag]               = e_tag_counts.get(tag, 0) + 1
            
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts
    
def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    
    pi = {(0, START_TAG, START_TAG): 1.0}
    bp = {}
    tags = e_tag_counts.keys() + [START_TAG]
    
    def q_prob(w, u, v):
        prob = q_cache.get((w, u, v))
        if prob is not None:
            return prob
            
        tri_prob = q_tri_counts.get((w, u, v), 0) / float(q_bi_counts.get((w, u), 0)) if q_bi_counts.get((w, u), 0) else 0.0
        bi_prob  = q_bi_counts.get((u, v), 0) / float(q_uni_counts.get(v, 0)) if q_uni_counts.get(v, 0) else 0.0
        uni_prob = q_uni_counts.get(v, 0) / float(total_tokens)
        
        prob = lambda1 * tri_prob + lambda2 * bi_prob + lambda3 * uni_prob
        q_cache[(w, u, v)] = prob
        return prob
    
    def tags_for_word(word, tags, e_word_tag_counts):
        ts = tags_for_word_cache.get(word)
        if ts is not None:
            return ts
        
        if word == START_TOKEN:
            ts = [START_TAG]
        else:
            ts = [t for t in tags if e_word_tag_counts.get((word, t), 0) > 0]
        
        tags_for_word_cache[word] = ts
        return ts
    
    # Pruning is done by calculating only for the tags that has emission probability > 0 for the current word.        
    for i, word in enumerate(sent):
        j = i + 1
        for u in tags_for_word(sent[i - 1] if i - 1 >= 0 else START_TOKEN, tags, e_word_tag_counts):
            for v in tags_for_word(word, tags, e_word_tag_counts):
                e       = float(e_word_tag_counts.get((word, v), 0)) / e_tag_counts[v]
                w_tags  = tags_for_word(sent[i - 2] if i - 2 >= 0 else START_TOKEN, tags, e_word_tag_counts)
                qs      = [q_prob(w, u, v) for w in w_tags]
                pis     = [pi.get((j - 1, w, u), 0) * q * e for w, q in zip(w_tags, qs)]
                max_value = max(pis)
                max_index = pis.index(max_value)
                pi[(j, u, v)] = max_value
                bp[(j, u, v)] = w_tags[max_index]
    
    last_max = 0
    for u in tags:
        for v in tags:
            prob = pi.get((len(sent), u, v), 0) * q_prob(u, v, STOP_TAG)
            if prob > last_max:
                last_max = prob
                if len(sent) > 0:
                    predicted_tags[-1] = v
                if len(sent) > 1:
                    predicted_tags[-2] = u
    
    if last_max == 0:
        return predicted_tags
    for i in range(len(sent) - 2)[::-1]:
        predicted_tags[i] = bp[(i + 3, predicted_tags[i + 1], predicted_tags[i + 2])]
    
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0.0
    
    success     = 0.0
    total_tags  = 0
    
    # Reset caches for new tests.
    global q_cache, tags_for_word_cache
    q_cache             = {}
    tags_for_word_cache = {}

    ### YOUR CODE HERE
    for i, sent in enumerate(test_data):
        #if i % 100 == 0:
        #    print i
        words = [w for w, t in sent]
        tags  = [t for w, t in sent]
        predicted_tags = hmm_viterbi(words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
        for guess, real in zip(predicted_tags, tags):
            if guess == real:
                success += 1
        total_tags += len(sent)
    acc_viterbi = success / total_tags
    
    ### END YOUR CODE
    
    return acc_viterbi

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    
    best_acc        = 0
    best_lambda1    = 0
    best_lambda2    = 0
    best_lambda3    = 0
    
    if TRAIN:
        print "#Performing grid search on lambda values..."
        delta = 0.05
        lambda1_vals = np.arange(0, 1, delta)
        lambda2_vals = np.arange(0, 1, delta)

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
                    lambda1 = l1val
                    lambda2 = l2val
                    lambda3 = 1 - (lambda1 + lambda2)
                    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
                    if acc_viterbi > best_acc:
                        best_acc = acc_viterbi
                        best_lambda1, best_lambda2, best_lambda3 = lambda1, lambda2, lambda3
                    print ('%.4f' % acc_viterbi),
            print
        
        print
        

        lambda1 = best_lambda1
        lambda2 = best_lambda2
        lambda3 = best_lambda3

        print "found best lambda: {} {} {}.".format(lambda1, lambda2, lambda3)
        print "dev: acc hmm viterbi: " + str(best_acc)

    else:
        lambda1 = 0.9
        lambda2 = 0
        lambda3 = 0.1
        acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts) 
        print "dev: acc hmm viterbi: " + str(acc_viterbi)

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi