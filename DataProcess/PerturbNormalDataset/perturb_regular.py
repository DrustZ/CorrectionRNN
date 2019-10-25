# author : Mingrui Ray Zhang
from random import randint, choice
#pip install pattern
import pattern.en as pe

'''
Word deformation:
pertube a word into another grammar form (to simulate the grammar error)
such as verb tense: bring -> brought
'''
class PerturbRegular(object):
    def perturb(self, word, tag):
        res = ""
        # pertube verb
        if 'V' in tag:
            vs = pe.lexeme(word)
            res = choice(vs)

            while (res == word or len(res) > len(word)) and (vs[0] != word):
                res = choice(vs)
            if vs[0] == word:
                res = vs[1]

        #pertube plural/singlar noun
        if 'NNS' == tag:
            res = pe.singularize(word)
            if res == word:
                res = word[:-1]

        if len(res) > 0:
            return (res, word, (0, len(res)))
        else:
            #if the perturbed result is empty, we just randomly remove some chars in the word 
            removeLen = randint(1, min(len(word)-1, 3))
            lenw = len(word)
            removestart = lenw-removeLen
            return (word[:removestart]+word[removestart+removeLen:], word, (0, lenw-removeLen))

'''
P = PerturbRegular()
print (P.perturb("broken", "V"))
print (P.perturb("selves", "NNS"))
print (P.perturb("plays", "V"))
'''
