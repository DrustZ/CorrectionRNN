#coding:utf-8
# author : Mingrui Ray Zhang
#requirement: nltk
from perturb_ConceptNet_synonym import *
from perturb_qwerty import *
from perturb_regular import *
from random import randint, choice
from nltk.corpus import stopwords
import nltk
import string

'''
    accepts a sentence (or word sequence), and generate perturbated sentece
    perturbated:
        1. randomly change a word by simulating QWERTY typing
        2. randomly drop a word / drop stopword
        3. randomly delete chars in a word (last chars), ("s", "es" - delete) / verb - random tense
        4. randomly change a word to its synonym
'''
class PerturbGenerator(object):
    """docstring for PerturbGenerator"""
    def __init__(self):
        self.pert_synonym = PerturbSynonym()
        self.pert_delete = PerturbRegular()
        self.pert_qwerty = PerturbQWERTY()
        self.stopwords = set(stopwords.words('english'))
        # print ("load over...")
        # need to download first time
        # nltk.download('punkt')
        # nltk.download('stopwords')
        # nltk.download('averaged_perceptron_tagger')

    # get the idx'th token's span in the original text
    def spans(self, tokens, txt, idx):
        offset = 0
        for i in range(len(tokens)):
            token = tokens[i]
            offset = txt.find(token, offset)
            if i == idx:
                return (offset, offset+len(token))
            offset += len(token)

    # return (string_with_mistake, correction, range_in_string:(start, length), whether to add an extra whitespace(0 - no, 1- before, 2-after))
    def perturb(self, stringseq, method=0):
        assert len(stringseq) > 0, "sentence must be more than one char!"
        assert any(c.lower() in string.ascii_lowercase for c in stringseq), "the sentence has to contain at least one alphabet letter!"
        tokens=nltk.wordpunct_tokenize(stringseq)
        # print (tokens)
        #if method is not specified, we randomly select one.
        if method == 0:
            method = randint(1,4)
        elif method < 0:
            method = randint(1,4+method) # eliminate certain methods
        
        # qwerty
        if method == 1:
            while True:
                randidx = randint(0, len(tokens)-1)
                token = tokens[randidx]
                #has to be at least one alpha letter
                if not any(c.lower() in string.ascii_lowercase for c in token):
                    continue
                res = self.pert_qwerty.perturb(token)
                if res[0] == res[1].lower():
                    continue
                fr, to = self.spans(tokens, stringseq, randidx)
                return (stringseq[:fr]+res[0]+stringseq[to:], res[1], (fr+res[2][0], res[2][1]), 0)
        # drop
        elif method == 2:

            stopwords = []
            for i, token in enumerate(tokens):
                if token in self.stopwords:
                    stopwords.append(i)

            if len(stopwords) > 0:
                randidx = choice(stopwords)
                token = tokens[randidx]
            else:
                randidx = randint(0, len(tokens)-2)
                token = tokens[randidx]

            fr, to = self.spans(tokens, stringseq, randidx)
            # remove a whitespace if drop the word
            if to < len(stringseq) and stringseq[to] == ' ':
                # trailing whitespace
                return (stringseq[:fr]+stringseq[to+1:], tokens[randidx], (fr, 0), 2 )
            elif stringseq[fr-1] == ' ':
                # leading whitespace
                return (stringseq[:fr-1]+stringseq[to+1:], tokens[randidx], (fr-1, 0), 1 )
            else:
                return (stringseq[:fr]+stringseq[to+1:], tokens[randidx], (fr, 0), 0 )
        # delete chars
        elif method == 3:
            pos_tags = nltk.pos_tag(tokens)
            cnt = 0
            while True:
                cnt += 1
                if cnt > 5:
                    break
                randidx = randint(0, len(tokens)-2)
                token = tokens[randidx]
                if not any(c.lower() in string.ascii_lowercase for c in token):
                    continue
                if len(token) < 2 or pos_tags[randidx] in ["TO", "SYM"]:
                    continue
                res = self.pert_delete.perturb(token, pos_tags[randidx])
                fr, to = self.spans(tokens, stringseq, randidx)
                return (stringseq[:fr]+res[0]+stringseq[to:], res[1], (fr+res[2][0], res[2][1]), 0 )
            return self.perturb(stringseq, -2)
        # synonym
        elif method == 4:
            cnt = 0
            while True:
                cnt += 1
                if cnt > 5:
                    # in case of dead loop
                    break
                randidx = randint(0, len(tokens)-2)
                token = tokens[randidx]
                # has to be a word
                if not all(c.lower() in string.ascii_lowercase for c in token) :
                    continue
                res = self.pert_synonym.perturb(token)
                if res[0] == res[1]:
                    continue
                fr, to = self.spans(tokens, stringseq, randidx)
                return (stringseq[:fr]+res[0]+stringseq[to:], res[1], (fr, res[2][1]), 0)
            return self.perturb(stringseq, -1)

        