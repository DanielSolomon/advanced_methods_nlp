from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import numpy as np

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word
    features['next_word'] = next_word
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['prev_tag_prevprev_tag'] = prev_tag + " " + prevprev_tag
    features['prev_word_tag'] = prev_word + " " + prev_tag
    features['prevprev_word_tag'] = prevprev_word + " " + prevprev_tag
    
    # capitalization features
    features['all_caps'] = prev_tag != '*' and curr_word == curr_word.upper()
    features['starts_with_cap'] = prev_tag != '*' and curr_word[0].isupper() 

    # prefix / suffix features
    for pref in ('re', 'de', 'dis'):
        features['pref_'+pref] = int(curr_word.lower().startswith(pref))
    for suff in ('ed', 's', 'es', 'ing', 'al', 'ed', 'ly'):
        features['suff_'+suff] = int(curr_word.lower().endswith(suff))
        
    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    return examples, labels
    print "done"

def memm_greeedy(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    for i in xrange(len(sent)):
        if i == 0:
            prevprev_tag = '*'
            prev_tag = '*'
        elif i == 1:
            prevprev_tag = '*'
            prev_tag = predicted_tags[i-1]
        else:
            prevprev_tag = predicted_tags[i-2]
            prev_tag = predicted_tags[i-1]
            
        # build vectorized features
        if i > 1:
            prevprev_word = sent[i - 2]
        else:
            prevprev_word = '<s>'
        if i > 0:
            prev_word = sent[i - 1]
        else:
            prev_word = '<s>'
        if i < (len(sent) - 1):
            next_word = sent[i + 1]
        else:
            next_word = '</s>'
            
        if (sent[i], next_word, prev_word, prevprev_word, prev_tag, prevprev_tag) in features_cache:
            vectorized_features = features_cache[(sent[i], next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)]
        else:
            features = extract_features_base(sent[i], next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)
            vectorized_features = vectorize_features(vec, features)
            features_cache[(sent[i], next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)] = vectorized_features
        
        predicted_tags[i] = index_to_tag_dict[logreg.predict(vectorized_features)[0]]
            
    return predicted_tags

    
def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    all_tags = tagset.keys()
    prob_dict = {(-1,'*','*') : (1, None)}
    for k in xrange(len(sent)):
        u_tags = None
        
        if k == 0:
            u_tags = {'*'}
        else:
            if sent[k-1] in tagsOfWord:
                 u_tags = tagsOfWord[sent[k-1]]
            else:
                u_tags = all_tags

        if k > 1:
            prevprev_word = sent[k - 2]
        else:
            prevprev_word = '<s>'
        if k > 0:
            prev_word = sent[k - 1]
        else:
            prev_word = '<s>'
        if k < (len(sent) - 1):
            next_word = sent[k + 1]
        else:
            next_word = '</s>'
                
        for u_tag in u_tags:
            v_tag = [""] * len(tagset)
            v_prob = [0] * len(tagset)
            w_tags = None
            
            if k <= 1:
                # not a word
                w_tags = {'*'}
            else:
                if sent[k-2] in tagsOfWord:
                    w_tags = tagsOfWord[sent[k-2]]
                else:
                    u_tags = all_tags = all_tags
            
            for w_tag in w_tags:
                
                if (sent[k], next_word, prev_word, prevprev_word, u_tag, w_tag) in predictions_cache:
                    # get the probability for this prediction from the cache
                    pr = predictions_cache[(sent[k], next_word, prev_word, prevprev_word, u_tag, w_tag)]
                else:
                    # no probability in cache so compute features
                    features = extract_features_base(sent[k], next_word, prev_word, prevprev_word, u_tag, w_tag)
                    pr = np.asarray(logreg.predict_proba(vectorize_features(vec, features))[0])
                    predictions_cache[vectorize_features(vec, features)] = pr                
                pr_vec = prob_dict[(k-1, w_tag, u_tag)][0] * pr
                for i in xrange(len(tagset)):
                    if pr_vec[i] > v_prob[i]:
                        v_tag[i] = w_tag
                        v_prob[i] = pr_vec[i]
            
            for i in xrange(len(tagset)):
                prob_dict[(k, u_tag, index_to_tag_dict[i])] = (v_prob[i], v_tag[i])

    # copute predicted_tags
    if len(sent) <= 1: all_tags = {'*'}
    max_prob = 0
    for v in all_tags:
        for u in all_tags:
            if (len(sent)-1, u, v) not in prob_dict:
                continue
            if  prob_dict[(len(sent)-1, u, v)][0] > max_prob:
                max_prob =  prob_dict[(len(sent)-1, u, v)][0]
                predicted_tags[len(sent)-1], predicted_tags[len(sent)-2] = v, u

    # compute the MC backwards
    for k in range(len(sent)-3,-1,-1):
        predicted_tags[k] = prob_dict[(k+2, predicted_tags[k+1], predicted_tags[k+2])][1]
    
    ### END YOUR CODE
    return predicted_tags

def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    ### YOUR CODE HERE
    all_count = 0
    greedy_success = 0
    viterbi_success = 0
    for sent in test_data:
        words = [s[0] for s in sent]
        tags = [s[1] for s in sent]
        greedy_res = memm_greeedy(words, logreg, vec)
        viterbi_res = memm_viterbi(words, logreg, vec)
        for i in xrange(len(sent)):
            if tags[i] == greedy_res[i]:
                greedy_success += 1 
            if tags[i] == viterbi_res[i]:
                viterbi_success += 1
            else:
                # print out errors
                print "Error (Viterbi) in the sentence '%s':" % ' '.join(words)
                print "'%s' should be %s but tagged as %s" % (sent[i][0], sent[i][1], viterbi_res[i])
                print '===='
            
        all_count += len(sent)
    
    acc_greedy = float(greedy_success) / all_count
    acc_viterbi = float(viterbi_success) / all_count
    ### END YOUR CODE
    return acc_viterbi, acc_greedy

if __name__ == "__main__":

    features_cache = {}
    predictions_cache = {}

    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    #The log-linear model training.
    #NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1
    
    # this is a dictionary containing a set of possible tags for each word
    tagsOfWord = {}
    for train_sent in train_sents:
        for word,tag in train_sent:
            if word not in tagsOfWord: tagsOfWord[word] = set()
            tagsOfWord[word].add(tag)           
    
    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=156, solver='lbfgs', C=100000, verbose=1, n_jobs=-1)
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "done, " + str(end - start) + " sec"
    #End of log linear model training
    
    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + str(acc_greedy)
    print "dev: acc memm viterbi: " + str(acc_viterbi)
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi