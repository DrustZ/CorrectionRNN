# author : Mingrui Ray Zhang
from PerturbationGenerator import PerturbGenerator as PG
import sys
import csv
import string
from random import randint
from tqdm import tqdm
from nltk import sent_tokenize
csv.field_size_limit(sys.maxsize)

P = PG()

table = str.maketrans({key: ' ' for key in string.punctuation})

# this file purturbs the normal dataset such as yelp review into 
# training dataset by injecting errors
# The output format will be error_text+space+correction+\t+output_five_word
# such as : have a god day good\thave a good day eos
# the output format details is explained in our paper (middle word as correction)
# output 5 sequence label
def perturbFile_out5(fname):
    with open(fname) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(tqdm(reader, leave=False)):
            text = ""
            for tx in line[1:]:
                sents = sent_tokenize(tx)
                for sent in sents:
                    sent = sent.replace('\n', ' ').replace('\a', ' ').replace('\\', '').strip()
                    sent = sent.translate(table)
                    if (len(sent) == 0) or (not any(c.lower() in string.ascii_lowercase for c in sent)):
                        continue
                    try:
                        if 5 <= len(sent.split()) <= 20:
                            res = P.perturb(sent)
                            errorsent, correction, crange, space = res

                            if space > 0:
                                errorsent = errorsent[:crange[0]]+' '+errorsent[crange[0]:]

                            start_idx = max(0, crange[0]-1)
                            while start_idx > 0 and errorsent[start_idx] != ' ':
                                start_idx -= 1

                            end_idx = crange[0]+crange[1]
                            while end_idx < len(errorsent) and errorsent[end_idx] != ' ':
                                end_idx += 1

                            labelpre = errorsent[:start_idx].split()
                            labelafter = errorsent[end_idx:].split()

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
                            label = labelpre 
                            if start_idx < crange[0]:
                                label += ' '+errorsent[start_idx:crange[0]]+correction
                            else:
                                label += ' '+correction
                            if end_idx > crange[0]+crange[1]:
                                label += errorsent[crange[0]+crange[1]:end_idx] + ' ' + labelafter
                            else:
                                label += ' ' + labelafter
                            print ("%s %s\t%s" % (' '.join(errorsent.split()), correction, ' '.join(label.split())))
                    except:
                        continue

# perturbFile_out5("/datasets/amazon_review_full_csv/test.csv")