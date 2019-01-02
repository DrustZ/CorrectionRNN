from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

import os
import math
import argparse
import torch
from model import *
from tqdm import tqdm
from torch import optim
from Dataset import MyDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
import torch.backends.cudnn as cudnn
from nltk.tokenize import wordpunct_tokenize

PAD_TOKEN = 0
SOS_token = 1
EOS_token = 2

mlength = 20

def eval_randomly(input_seq, test_dataset, encoder, decoder, max_length=mlength):
    input_seq = wordpunct_tokenize(input_seq)
    input_seqs = [test_dataset.indexes_from_sentence_char_to_word(input_seq)]
    input_lengths = [len(s) for s in input_seqs]
    input_batches = torch.LongTensor(input_seqs).transpose(0, 1)
    max_length = max(max_length, input_lengths[0])
    input_batches = input_batches.cuda()
    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Run through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        # decoder_input = torch.LongTensor([SOS_token]) # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
        # decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = [ ([SOS_token], 0, decoder_hidden) ]
        # decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        # Run through decoder
        for di in range(5):
            dlen = len(decoded_words)
            for i in range(dlen):
                decoder_hidden = decoded_words[i][2]
                decoder_input = torch.LongTensor([decoded_words[i][0][-1]]).cuda()
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

                # Choose top word from output

                topv, topi = decoder_output.data.topk(2)
                for j in range(len(topi[0])):
                    ni = topi[0][j].item()
                    decoded_words.append( (decoded_words[i][0]+[ni], decoded_words[i][1]+topv[0][j].item(), decoder_hidden) )

            decoded_words = decoded_words[dlen:]
            decoded_words.sort(key=lambda item: -item[1])
            decoded_words = decoded_words[:2]

        decoded_words = decoded_words[0][0][1:]
        final_res = []
        for ni in decoded_words:
            if ni == EOS_token:
                final_res.append('<EOS>')
                # break
            else:
                final_res.append(test_dataset.index2word[ni])

    return (' '.join(final_res))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Correction classifier')
    parser.add_argument('--train', '-t', required=True, type=str, help='file to train')
    parser.add_argument('--test', type=str, help='file to test')
    parser.add_argument('--load_en', type=str, help='encoder model to resume')
    parser.add_argument('--load_de', type=str, help='decoder model to resume')
    parser.add_argument('--max_length', type=int, default=20, help='max word length')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--n_layers', type=int, default=2, help="number of layers of GRU")
    parser.add_argument('--hidden_size', type=int, default=300, help="hidden size")
    global args
    args = parser.parse_args()
    return args    

def setup():
    eprint("loading...")
    cudnn.benchmark = True
    test_dataset = MyDataset(args.train, filter_pair=True, max_length = mlength, min_length = 3, max_word_length=args.max_length)
    voc_size = test_dataset.n_words
    
    encoder = C2WEncoderRNN(args.hidden_size, args.n_layers, dropout=args.dropout)
    decoder = BahdanauAttnDecoderRNN(voc_size, args.hidden_size, args.n_layers, dropout=args.dropout)
    eprint(encoder)
    eprint(decoder)
    eprint ("vocab size", voc_size)
    encoder.cuda()
    decoder.cuda()    

    if args.load_en and args.load_de:
        state_en = torch.load(args.load_en)
        state_de = torch.load(args.load_de)
        encoder.load_state_dict(state_en)
        decoder.load_state_dict(state_de)
        eprint('Loading parameters from {} {}'.format(args.load_en, args.load_de))

    return encoder, decoder, test_dataset

def equal_pre(label, pre, lword):
    labels = label.split()
    pres = pre.split()

    if labels[2].lower() != pres[2].lower() and labels[2].lower() != lword.lower():
        return False
    labels, pres = labels[:2]+labels[3:], pres[:2]+pres[3:]
    cnt = 0

    for i in range(len(labels)):
        if labels[i].lower() == pres[i].lower():
            cnt += 1
    if cnt >= 2:
        return True
    return False

def collate(batch):
    # Pad input with the PAD symbol
    def pad_seq_word(seq, max_length):
        seq += [[PAD_TOKEN for _ in range(args.max_length)] for _ in range(max_length - len(seq))]
        return seq

    # Pad target with the PAD symbol
    def pad_seq(seq, max_length):
        seq += [PAD_TOKEN for i in range(max_length - len(seq))]
        return seq

    batch = list(zip(*batch))
    seq_pairs = sorted(zip(batch[0], batch[1]), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq_word(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_padded).transpose(0, 1)
    target_var = torch.LongTensor(target_padded).transpose(0, 1)

    return (input_var, input_lengths, target_var, target_lengths)

def collate_fn():
    return lambda batch: collate(batch)

def correct(output, target, target_lengths):
    target = target.transpose(0,1).float()
    output = output.transpose(0,1)
    acc = 0
    for i in range(len(target_lengths)):
        acc += target[i, :target_lengths[i]].eq(output[i, :target_lengths[i]]).float().mean()
    return acc

def evaluate(test_data, encoder, decoder):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    accuracy = 0
    cnt = 0

    with torch.no_grad():
        for i, (src, len_src, trg, len_trg) in enumerate(tqdm(test_data, leave=False)):
            src = src.cuda()
            trg = trg.cuda()
            batch_size = src.shape[1]
            # Run words through encoder
            encoder_outputs, encoder_hidden = encoder(src, len_src, None)
            # Prepare input and output variables
            decoder_input = torch.LongTensor([SOS_token] * batch_size)
            decoder_input = decoder_input.cuda()
            decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
            
            max_target_length = max(len_trg)
            all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)
            all_decoder_outputs = all_decoder_outputs.cuda()

            top1s = torch.zeros(max_target_length, batch_size)
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_output, decoder_hidden, decoder_attn = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                all_decoder_outputs[t] = decoder_output
                top1 = decoder_output.max(1)[1]
                top1s[t] = top1.cpu()
                decoder_input = top1

            loss = F.nll_loss(all_decoder_outputs.view(-1, decoder.output_size),
                               trg.contiguous().view(-1),
                               ignore_index=PAD_TOKEN)
            
            cnt += int(src.shape[1])
            total_loss += float(loss.item())

            accuracy += correct(top1s, trg.cpu(), len_trg)

    return total_loss / cnt, accuracy / cnt


def evalFileBatch():
    args = parse_arguments()
    encoder, decoder, dataset = setup()
    test_data = MyDataset(args.train, filter_pair=True, max_length = mlength, min_length = 3, max_word_length=args.max_length, train=False)
    test_data = DataLoader(test_data, batch_size=32, pin_memory=True,
                            shuffle=False, num_workers=2, collate_fn=collate_fn())
    loss, acc = evaluate(test_data, encoder, decoder)
    print ("acc:", acc, "loss", loss)

def evalFile(fname, encoder, decoder, dataset):
    cnt = 0
    right = 0
    with open(fname) as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, leave=False)):
            sent, label = line.strip().split("\t")
            lword = sent.split()[-1]
            # sent = sent[2:]
            if len(sent.split()) <= 3:
                continue
            cnt += 1
            pre = eval_randomly(sent, dataset, encoder, decoder)
            if not equal_pre(label, pre, lword):
                print ("[wrong] %s\n[lb] %s\n[pr] %s" % (sent, label, pre))
            else:
                # print ("[right] %s\n[lb] %s\n[pr] %s" % (sent, label, pre))
                right += 1
    print ("acc :", right/cnt)

if __name__ == "__main__":
    try:
        args = parse_arguments()
        encoder, decoder, dataset = setup()
        # evalFile("/media/ray/My_Passport/ubuntu/codes/DragCorrect/DatasetProcessing/conll5", encoder, decoder, dataset)
        # evalFile("wiki.validate5", encoder, decoder, dataset)
        evalFile("testexp", encoder, decoder, dataset)
        # evalFileBatch("testdata_eval")
        # evalFile("/media/ray/My_Passport/ubuntu/codes/DragCorrect/DatasetProcessing/conll5")
        
        # while True:
        #     sent = input('Enter a sent: [no punc, cor at last]; "quit" to quit\n')
        #     if sent == "quit":
        #         break
        #     print (eval_randomly(sent.strip(), dataset, encoder, decoder))
    except KeyboardInterrupt as e:
        eprint("[STOP]", e)
