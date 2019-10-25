# -*- coding: utf-8 -*-
# author : Mingrui Ray Zhang
from bs4 import BeautifulSoup
import string
import nltk
import random

"""
To convert the conll (13/14) datset to training set
please execute this script using the conll_outputfile() function
and use > symbol to feed the output into a new file
"""

class CoNLLdoc(object):
    """
    The class that parse CoNLL dataset into training dataset
    It will locate the error sentence in the dataset, and 
    randomly extract from 25-35 character length including 
    the error. Then print out the training-set format.

    Example:
    If a correction happens in the dataset, like 
    'I used to go to schools by bus' and 
    the correction is 'schoos' -> 'school'
    then it will randomly select 25-35 chars around the error,
    such as 'to go to schools by bus'. Then it will output 
    the format for training, which is 
    (error+correction, output (five words)):
    'to go to schools by bus school, go to school by bus'

    """
    def __init__(self):
        self.paras = []
        self.mistakes = []
        self.sentences = []
        self.sentence_starts = []
        self.table = str.maketrans('', '', string.punctuation)

    def setPara(self, paras):
        self.paras = paras

    def setMistakes(self, mists):
        self.mistakes = mists

    def getCorrectionInstance(self, idx):
        if idx >= len(self.mistakes):
            return ""
        mistake = self.mistakes[idx]
        if int(mistake["start_par"]) != int(mistake["end_par"]):
            return ""
        para = self.paras[int(mistake["start_par"])]
        return para[int(mistake["start_off"]):int(mistake["end_off"])] + " -> " + mistake["correction"]

    def printAllCorrections(self):
        for i in range(len(self.mistakes)):
            print (self.getCorrectionInstance(i))

    def printValidCorrection_label5(self, randomize=True):
        for i in range(len(self.mistakes)):
            mistake = self.mistakes[i]
            if int(mistake["start_par"]) != int(mistake["end_par"]):
                continue
            paraidx = int(mistake["start_par"])
            para = self.paras[paraidx]
            offsetstart = int(mistake["start_off"])
            offsetend = int(mistake["end_off"])
            if mistake["correction"] == None or len(mistake["correction"].strip()) == 0 or len(mistake["correction"].split()) > 1:
                continue
            correction = mistake["correction"].translate(self.table).lower()
            if len(correction) == 0:
                continue

            if randomize:
                startidx = max(0, offsetstart - random.randint(25, 35))
                endidx = min(offsetend+random.randint(25,35), len(para))
            else:
                startidx = max(0, offsetstart-35)
                endidx = min(offsetend+35, len(para))
                
            if startidx > 0:
                startidx = para[startidx:].find(' ')+startidx+1
            if endidx < len(para):
                endidx = para[:endidx].rfind(' ')
            if endidx < offsetend:
                continue
            sent = para[startidx:endidx]
            offsetstart -= startidx
            offsetend -= startidx

            present = sent[:offsetstart].translate(self.table)
            aftersent = sent[offsetend:].translate(self.table)
            error = sent[offsetstart:offsetend].translate(self.table).lower()
            if error.strip() == correction.strip() or len(error.split()) > 1:
                continue

            labelpre = sent[:offsetstart].translate(self.table).split()
            labelafter = sent[offsetend:].translate(self.table).split()

            if len(labelpre) >= 2:
                labelpre = labelpre[-2:]
            elif len(labelpre) == 0:
                labelpre = ["BOS", "BOS"]
            else:
                labelpre = ["BOS"] + labelpre
            if len(labelafter) >= 2:
                labelafter = labelafter[:2]
            elif len(labelafter) == 0:
                labelafter = ["EOS", "EOS"]
            else:
                labelafter = labelafter + ["EOS"]
            labelpre = ' '.join(labelpre)
            labelafter = ' '.join(labelafter)
            labelpre = labelpre.lower()
            labelafter = labelafter.lower()
            correction = correction.lower()

            print ("%s %s\t%s" % (sent.translate(self.table).lower(), correction.lower(), labelpre+' '+correction+' '+labelafter))

def parseDocs(fname):
    res = []
    with open(fname) as f:
        content = f.read()
        soup = BeautifulSoup(content)
        docs = soup.find_all('doc')
        for doc in docs:
            conlldoc = CoNLLdoc()
            paras = []
            if doc.find("title") != None:
                paras.append(doc.title.string.strip())
            for p in doc.find_all("p"):
                if p.string != None:
                    paras.append(p.string.strip()) 
                else:
                    paras.append("")

            corrections = []
            for mistake in doc.find_all("mistake"):
                item = {}
                item["start_par"]  =  mistake['start_par']
                item["start_off"]  =  mistake['start_off']
                item["end_par"]    =  mistake['end_par']
                item["end_off"]    =  mistake['end_off']
                item["ctype"]      =  mistake.type.string.strip()
                if mistake.correction.string != None:
                    item["correction"] =  mistake.correction.string.strip()
                else:
                    item["correction"] = ''
                corrections.append(item)
            conlldoc.setPara(paras)
            conlldoc.setMistakes(corrections)
            res.append(conlldoc)
    return res

#fname: the conll dataset name,
# such as /conll13/official.sgml
def conll_outputfile(fname):
    texts = parseDocs(fname)
    for text in texts:
        text.printValidCorrection_label5(False)

#conll_outputfile("/path/to/conll/dataset")