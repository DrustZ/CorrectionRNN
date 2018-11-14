import numpy as np
from torch.utils.data import Dataset
import unicodedata
import string
import random
import math
import torch
from nltk.tokenize import wordpunct_tokenize

class MyDataset(Dataset):
    def __init__(self, data_path, filter_pair=True, max_length = 20, min_length = 3, max_word_length=20, train=True):
        self.data_path = data_path
        # self.vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        # no punc:
        self.vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789""")
        self.char_dic = {}
        for i in range(len(self.vocabulary)):
            self.char_dic[self.vocabulary[i]] = i+3 #0 for pad, 1 for start , 2 for end

        self.word2index = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens
        self.MAX_LENGTH = max_length
        self.MIN_LENGTH = min_length
        self.max_word_length = max_word_length
        self.EOS_token = 2

        lines = open(data_path).read().strip().split('\n')
        # Split every line into pairs and normalize
        self.pairs = [[s for s in l.split('\t')[:2]] for l in lines]
        print("Read %d sentence pairs" % len(self.pairs))
        
        if filter_pair:
            self.filter_pairs()
            print("Filtered to %d pairs" % len(self.pairs))
        self.length = len(self.pairs)
        
        print("Indexing words...")
        for i in range(self.length):
            self.pairs[i] = wordpunct_tokenize(self.pairs[i][0]), wordpunct_tokenize(self.pairs[i][1])
            for word in self.pairs[i][1]:
                self.index_word(word)

        if train:
            lenp = len(self.pairs)
            self.pairs = self.pairs[:int(lenp*0.95)]
        else:
            lenp = len(self.pairs)
            self.pairs = self.pairs[-int(lenp*0.05):]
        self.length = len(self.pairs)

        print('Indexed %d words in output' % (self.n_words))

    def filter_pairs(self):
        filtered_pairs = []
        for pair in self.pairs:
            if len(pair[0].split()) >= self.MIN_LENGTH and len(wordpunct_tokenize(pair[0])) <= self.MAX_LENGTH:
                if all([len(token) <= self.max_word_length-2 for token in pair[0]]):
                    filtered_pairs.append(pair)
        self.pairs = filtered_pairs

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def indexes_from_sentence_char_to_word(self, sentence):
        sent_vec = []
        for word in sentence:
            vec = [self.char_dic[i] for i in word if i in self.char_dic]
            vec.insert(0, 1) #start_of_word mark
            vec.append(2) # end_of_word mark
            if len(vec) > self.max_word_length:
                 vec = vec[:self.max_word_length-1]+[2]
            elif 0 < len(vec) < self.max_word_length:
                vec += [0 for _ in range(self.max_word_length - len(vec))]
            elif len(vec) == 0:
                vec = [0 for _ in range(self.max_word_length)]
            sent_vec.append(vec)
        return sent_vec

    # Return a list of indexes, one for each word in the sentence, plus EOS
    def indexes_from_sentence_word(self, sentence):
        return [self.word2index[word] for word in sentence] + [self.EOS_token]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        errorsent, correctsent = self.pairs[index]
        errorsent = self.indexes_from_sentence_char_to_word(errorsent)
        correctsent = self.indexes_from_sentence_word(correctsent)
        return errorsent, correctsent
