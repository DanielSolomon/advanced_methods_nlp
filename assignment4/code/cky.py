from PCFG import PCFG
import collections, math

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents

def calculate_pcfg_rule_probs(pcfg):
    rules       = pcfg._rules
    sums        = pcfg._sums
    rule_probs  = collections.defaultdict(float)

    for key, derived in rules.items():
        for derive in derived:
            rule_probs[(key, tuple(derive[0]))] = derive[1] / sums[key]
    return rule_probs

def gentree_by_bp(sent, bp, i, j, r):
    rule, s     = bp[(i, j, r)]
    if s is None:
        return '({} {})'.format(rule[0], rule[1][0])
    left        = rule[1][0]
    right       = rule[1][1]
    left_tree   = gentree_by_bp(sent, bp, i, s, left)
    right_tree  = gentree_by_bp(sent, bp, s + 1, j, right)
    return '({} ({} {}))'.format(rule[0], left_tree, right_tree)
    
def cky(pcfg, sent):
    sent = sent.split()
    
    pi          = collections.defaultdict(float)
    rule_probs  = calculate_pcfg_rule_probs(pcfg)
    bp          = {}
        
    # base
    for i in range(len(sent)):
        for r in pcfg._rules.keys():
            pi[(i, i, r)] = rule_probs[(r, (sent[i],))]
            bp[(i, i, r)] = ((r, (sent[i],)), None)
            
    # recursion
    for l in range(1, len(sent)):
        for i in range(len(sent) - 1):
            j = i + l
            for r in pcfg._rules.keys():
                max_prob  = 0.0
                best_rule = None
                best_s    = -1
                for s in range(i, j):
                    for rule in pcfg._rules[r]:
                        rule    = tuple(rule[0])
                        if pcfg.is_preterminal(rule):
                            continue
                        i_s     = pi[(i, s, rule[0])]
                        s_j     = pi[(s + 1, j, rule[1])]
                        q_prob  = rule_probs[(r, rule)]  

                        prob    = i_s * s_j * q_prob
                        if prob > max_prob:
                            max_prob  = prob
                            best_rule = (r, rule)
                            best_s    = s
                pi[(i, j, r)] = max_prob
                bp[(i, j, r)] = best_rule, best_s
    
    if pi[(0, len(sent) - 1, 'S')] > 0:
        return gentree_by_bp(sent, bp, 0, len(sent) - 1, 'S')
    else:
        return "FAILED TO PARSE!"

if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
