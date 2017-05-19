import collections
from data import *

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    word_tag_map = dict()
    
    for sent in train_data:
        for word, tag in sent:
            if not word in word_tag_map:
                word_tag_map[word] = collections.Counter()
            word_tag_map[word].update([tag])
    
    for key, value in word_tag_map.iteritems():
        word_tag_map[key] = value.most_common(1)[0][0]
        
    return word_tag_map
    
def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    
    correct = 0.0
    total   = sum([len(sent) for sent in test_set])
    
    for sent in test_set:
        for word, tag in sent:
            if pred_tags.get(word) == tag:
                correct += 1
                
    return correct / total
    
    
if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " , most_frequent_eval(dev_sents, model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)