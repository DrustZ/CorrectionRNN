import os
import math
import argparse
import torch
import string

from model import *
from torch import optim
from Dataset import MyDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
import torch.backends.cudnn as cudnn

from difflib import SequenceMatcher
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import tornado.httpserver
import tornado.ioloop
import tornado.web
import json
import numpy as np
from tornado import gen
from time import time 

PAD_TOKEN = 0
SOS_token = 1
EOS_token = 2

MAX_W_LENGTH=20
N_LAYERS  = 2
HIDDEN_SIZE = 300
SET_DIR = "train_data5"#"small_train_data5_amazon"
LOAD_PREF = "dataset_freq2_"
EN_DIR = "best_en_5out_freq2.pth"#"best_en_5out_amazon.pth"
DE_DIR = "best_de_5out_freq2.pth"

ENCODER = None
DECODER = None
DATASET = None

lmtzr = WordNetLemmatizer()
table = str.maketrans({key: ' ' for key in string.punctuation})

def eval_sent(input_seq):
    input_seq = wordpunct_tokenize(input_seq)
    input_seqs = [DATASET.indexes_from_sentence_char_to_word(input_seq)]
    input_lengths = [len(s) for s in input_seqs]
    input_batches = torch.LongTensor(input_seqs).transpose(0, 1)
    input_batches = input_batches.cuda()
    # Set to not-training mode to disable dropout
    ENCODER.eval()
    DECODER.eval()
    with torch.no_grad():
        # Run through encoder
        encoder_outputs, encoder_hidden = ENCODER(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        decoder_hidden = encoder_hidden[:DECODER.n_layers] # Use last (forward) hidden state from encoder

        # Store output words and attention states
        decoded_words = [ ([SOS_token], 0, decoder_hidden) ]

        # Run through decoder
        for di in range(5):
            dlen = len(decoded_words)
            for i in range(dlen):
                decoder_hidden = decoded_words[i][2]
                decoder_input = torch.LongTensor([decoded_words[i][0][-1]]).cuda()
                decoder_output, decoder_hidden, decoder_attention = DECODER(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                topv, topi = decoder_output.data.topk(2)
                for j in range(len(topi[0])):
                    ni = topi[0][j].item()
                    decoded_words.append( (decoded_words[i][0]+[ni], decoded_words[i][1]+topv[0][j].item(), decoder_hidden) )

            decoded_words = decoded_words[dlen:]
            decoded_words.sort(key=lambda item: -item[1])
            decoded_words = decoded_words[:2]

        prob = float(decoded_words[0][1])
        decoded_words = decoded_words[0][0][1:]
        final_res = []
        for ni in decoded_words:
            if ni == EOS_token:
                final_res.append('<EOS>')
                # break
            else:
                final_res.append(DATASET.index2word[ni])

    return final_res, prob



def setup():
    global ENCODER, DECODER, DATASET
    cudnn.benchmark = True
    DATASET = MyDataset(SET_DIR, filter_pair=True, max_length = 20, min_length = 3, \
        max_word_length=MAX_W_LENGTH, load_fprefix=LOAD_PREF, onlylower=True, freq_threshold=2)
    voc_size = DATASET.n_words
    
    ENCODER = C2WEncoderRNN(HIDDEN_SIZE, N_LAYERS, dropout=0)
    DECODER = BahdanauAttnDecoderRNN(voc_size, HIDDEN_SIZE, N_LAYERS, dropout=0)
    ENCODER.cuda()
    DECODER.cuda()

    state_en = torch.load(EN_DIR)
    state_de = torch.load(DE_DIR)
    ENCODER.load_state_dict(state_en)
    DECODER.load_state_dict(state_de)
    print('Loading parameters from {} {}'.format(EN_DIR, DE_DIR))

#score for Edit Distance
def Score_EDIT(s1, s2):
    m = SequenceMatcher(None, s1, s2)
    return m.ratio()

def getLemmarScores(tokens, correction):
    res = []
    lcorrection_n = lmtzr.lemmatize(correction)
    lcorrection_v = lmtzr.lemmatize(correction, 'v')

    for token in tokens:
        ltoken_n = lmtzr.lemmatize(token)
        ltoken_v = lmtzr.lemmatize(token, 'v')
        if ltoken_n == lcorrection_n or ltoken_v == lcorrection_v:
            res.append(1)
        else:
            res.append(0)
    return np.array(res)

##server

class MainHandler(tornado.web.RequestHandler):
    def prepare(self):
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.json_args = json.loads(self.request.body)
        else:
            self.json_args = None

    def set_default_headers(self):
        # print ("setting headers!!!")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Content-Type', 'application/json; charset=UTF-8')

    def options(self):
        # no body
        self.set_status(204)
        self.finish()

    def get(self):
        self.write("Hello, world")

    @gen.coroutine
    def post(self):
        if self.json_args != None:
            if self.json_args["smart"] == 0:
                pos, lenc, _, prob, right_sent = yield self.getResponseOneLine(self.json_args["sent"], self.json_args["correction"])
                reply = json.dumps({
                    'start': pos[0],
                    'len' : lenc,
                    'prob': prob,
                    'sent': right_sent
                    })
            else:
                starts, ends, corrections = yield self.getResponseArray(self.json_args["sent"], self.json_args["correction"])
                reply = json.dumps({
                    'starts': starts,
                    'ends' : ends,
                    'corrections': corrections
                    })
            self.write(reply)
        else:
            self.write(json.dumps({'start': -1, 'len': -1, 'prob': -1}))

    
    @gen.coroutine
    def getResponseArray(self, text, correction):
        def getNonAlphaNumIdx(string):
            start = 0
            end = len(text)

            for i,c in enumerate(string):
                if not c.isalnum():
                    start = i
                    break

            for i in range(len(string)-1, -1, -1):
                if not string[i].isalnum():
                    end = i+1
                    break
            return start, end

        startidx = max(len(text)-60, 0)
        res, offsets = [], []
        sent = text[startidx:]
        start, end = getNonAlphaNumIdx(sent)
        if startidx == 0:
            start = 0

        res.append([sent[start:end], correction])
        offsets.append(start+startidx)
        # tmp = yield self.getResponseOneLine()
        # res.append([start+startidx+tmp[0][0], # realign start pos
        #     tmp[0][1], #cor length
        #     tmp[2], tmp[3]]) # correction, prob

        while startidx > 0:
            startidx = max(0, startidx-30)
            sent = text[startidx:startidx+60]
            start, end = getNonAlphaNumIdx(sent)
            if startidx == 0:
                start = 0
            res.append([sent[start:end], correction])
            offsets.append(start+startidx)
            # tmp = yield self.getResponseOneLine(sent[start:end], correction)
            # res.append([start+startidx+tmp[0][0], # realign start pos
            # tmp[0][1], #cor length
            # tmp[2], tmp[3]]) # correction, prob

        t = time()
        res = yield [self.getResponseOneLine(it[0], it[1]) for it in res]
        print ("multiple "+str(len(res)), time()-t)

        for i in range(len(res)):
            res[i] = [res[i][0][0]+offsets[i], # start 
                    res[i][0][1], res[i][2], res[i][3]] # length, correction, prob

        res.sort(key=lambda x : x[0], reverse=True)
        print (res)

        for i in range(len(res)-1, 0, -1):
            if res[i][0] == res[i-1][0]:
                if res[i][3] > res[i-1][3]:
                    res = res[:i-1]+res[i:]
                    continue
                else:
                    res = res[:i]+res[i+1:]
        
        starts = [i[0] for i in res]
        ends = [i[0]+i[1] for i in res]
        corrections = [i[2] for i in res]

        for i in range(len(starts)):
            if starts[i] == ends[i]:
                print (corrections[i] + ' ' + text[ends[i]:])
            else:
                print (corrections[i] + text[ends[i]:])
        print (starts)
        print (ends)
        return starts, ends, corrections 

    @gen.coroutine
    def getResponseOneLine(self, text, correction):
        def align_correction(tokens, res, correction):
            tokens = ["BOS", "BOS"] + tokens + ["EOS", "EOS"]
            most_idx = (-1, -1)
            max_cnt = 0

            tmpres = res[:2]+res[3:]

            for i in range(len(tokens)-4):
                compares1 = list(zip(tokens[i:i+4], tmpres)) # insert a new word
                cnt1 = sum([(pair[0] == pair[1]) for pair in compares1])
                if cnt1 > max_cnt:
                    max_cnt = cnt1
                    most_idx = (i, 0)
                
                if i+5 <= len(tokens):
                    compares2 = list(zip(tokens[i:i+2]+tokens[i+3:i+5], tmpres)) #change a exsiting word
                    cnt2 = sum([(pair[0] == pair[1]) for pair in compares2])
                    if cnt2 > max_cnt:
                        max_cnt = cnt2
                        most_idx = (i, 1)
            return most_idx
        #predict corrections here
        tokens = text.lower().translate(table).split()
        errsent = ' '.join(tokens)
        correction = ' '.join(correction.translate(table).split())

        token_offset = []
        offset = 0
        sub_score = []
        for i in range(len(tokens)):
            token = tokens[i]
            offset = text.lower().find(token, offset)
            token_offset.append(offset)
            offset += len(token)
            if token != correction:
                sub_score.append(Score_EDIT(token, correction))
            else:
                sub_score.append(0)
        sub_score = np.array(sub_score)

        lemma_score = getLemmarScores(tokens, correction)

        find = False
        #First, editdist
        if max(sub_score[1:-1]) >= 0.7:
            idx = sub_score[1:-1].argmax()+1
            pos = (token_offset[idx], len(tokens[idx]))
            if not tokens[idx].lower() == correction.lower():
                find = True
                right_sent = text[:pos[0]]+correction+text[pos[0]+pos[1]:]
                prob = 0
        #Second, lemmarization
        elif (lemma_score == 1).sum() == 1:
            idx = lemma_score.argmax()
            pos = (token_offset[idx], len(tokens[idx]))
            if not tokens[idx].lower() == correction.lower():
                find = True
                right_sent = text[:pos[0]]+correction+text[pos[0]+pos[1]:]
                prob = 0
        
        if not find:
            res, prob = eval_sent(errsent+' '+correction)

            if correction not in res[2]:
                res[2] = correction

            print ("res, prob", res, prob)

            crange = align_correction(tokens, res, correction)
            correction = res[2]
            if crange[1] == 0: #insert
                if crange[0] < len(tokens):
                    pos = (token_offset[crange[0]], 0)
                    right_sent = text[:pos[0]]+res[2]+' '+text[pos[0]:]
                else:
                    pos = (token_offset[-1]+len(tokens[-1]), 0)
                    right_sent = text[:pos[0]]+' '+res[2]+text[pos[0]:]
            else:
                pos = (token_offset[crange[0]], len(tokens[crange[0]]))
                right_sent = text[:pos[0]]+res[2]+text[pos[0]+pos[1]:]

        return [pos, len(correction), correction, math.exp(prob), right_sent]
        
def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8765)
    setup()
    print ("listen...")
    tornado.ioloop.IOLoop.current().start()

'''
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"smart":0,"sent":"this is the best thin I ve ever done", "correction":"thing"}' \
  10.0.0.55:8765
'''
