#coding:utf-8
# require: symspellpy
# author : Mingrui Ray Zhang

import numpy as np
import os
from random import randint, shuffle
from symspellpy.symspellpy import SymSpell, Verbosity  # import the module

'''
Perturb a word into a typo
by simulating the typing process of the word, introducing random bayesian noise (the keyboard spatial model)
The way to do perturbation is first simulate the typing process, 
and check if the output word can be spell checked to another word (this simulates when there's a typo and 
auto-correction did it wrongly)
If so, we use the wrongly suggestion as the error
else we just use the typo as the error
'''
class PerturbQWERTY(object):
    def __init__(self, max_edit_distance_dictionary=2, spell_fname="frequency_dic_en.txt", QWERTY_fname="QWERTYprobability"):
        # create object
        initial_capacity = 83000
        # maximum edit distance per dictionary precalculation
        prefix_length = 7
        self.sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,
                             prefix_length)
        # load dictionary
        dictionary_path = os.path.join(os.path.dirname(__file__),
                                       spell_fname)
        term_index = 0  # column of the term in the dictionary text file
        count_index = 1  # column of the term frequency in the dictionary text file
        if not self.sym_spell.load_dictionary(dictionary_path, term_index, count_index):
            print("Dictionary file not found")
            return

        #init QWERTY prob data 
        self.PROB = {}
        self.readProb(QWERTY_fname)
    
    def readProb(self, fname="QWERTYprobability"):
        probs = {}
        with open(fname) as f:
            for line in f:
                c1, _, c2, prob = line.split()
                if c1 not in probs:
                    probs[c1] = {}
                probs[c1][c2] = float(prob)
        res = {}
        #generate result proper for np.random.choince
        #probs['a'] = [ alt_chars: [a, q, w, s] 
        #               alt_probs: [.1, .1, .1, .1] ]
        for c1 in probs:
            acc_prob = 0
            if c1 not in res:
                res[c1] = [[], []]
            for c2 in probs[c1]:
                res[c1][0].append(c2)
                res[c1][1].append(probs[c1][c2])
                acc_prob += probs[c1][c2]
            for i,p in enumerate(res[c1][1]):
                res[c1][1][i] = p/acc_prob
            # res[c1][0].append(c1)
            # res[c1][1].append(1-acc_prob)
        self.PROB = res

    def checkword(self, input_term, max_edit_distance=2):
        suggestion_verbosity = Verbosity.TOP  # TOP, CLOSEST, ALL
        suggestions = self.sym_spell.lookup(input_term, suggestion_verbosity,
                                       max_edit_distance)
        if len(suggestions) > 0:
            return suggestions[0].term
        else:
            return input_term

    def perturb(self, word):
        # simulating the typing using spatial distribution
        # for example, given 'a', it might output 'q', 'w', 's', or 'z'
        def simulateTyping(letter):
            letter = letter.lower()
            if letter in self.PROB:
                return np.random.choice(self.PROB[letter][0], p=self.PROB[letter][1])
            else:
                return letter

        res, suggest_term = word.lower(), word
        
        rand_indices = list(range(len(word)))
        shuffle(rand_indices)
        #first of all, we make 1-3 typos in the letters
        type_errors = randint(1, min(3, len(word)))
        rand_indices = rand_indices[:type_errors]
        res = word
        for i in rand_indices:
            res = res[:i] + simulateTyping(res[i]) + res[i+1:]

        #after simulating the typo, we check if the spell-checker 
        #suggests the original word or another word (which could be the error)
        suggest_term = self.checkword(res)

        if suggest_term == word.lower():
            return (res, word, (0, len(res)))
        else: 
            return (suggest_term, word, (0, len(suggest_term)))

'''
test code
'''
'''
p = PerturbQWERTY()
for i in range(10):
    print (p.perturb("test"))
'''
