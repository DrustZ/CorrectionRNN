#coding:utf-8
# author : Mingrui Ray Zhang
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim.downloader as api
from random import randint

'''
pertube a word to its semantic similar word
using the glove twitter model
such as 'evening' -> 'night'
'''

class PerturbSynonym(object):
    def __init__(self):
        self.model = api.load("glove-twitter-100")

    def perturb(self, word):
        # if the word has similar words in the model
        try:
            res = self.model.most_similar(word.lower())
            idx = randint(0, min(4, len(res)-1))
            # print ("[SYN] get sim", word, res[idx][0])
            return (res[idx][0], word, (0, len(res[idx][0])))
        # else we just return the raw word
        except Exception as e:
            # print ("[SYN] no such word", word.lower(), str(e))
            return (word, word, (0, 0))
