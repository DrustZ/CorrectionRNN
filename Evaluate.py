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

def eval_randomly(input_seq, test_dataset, encoder, decoder, max_length=22):
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
        decoder_input = torch.LongTensor([SOS_token]) # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
        decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0].item()
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(test_dataset.index2word[ni])

            # Next input is chosen word
            decoder_input = torch.LongTensor([ni])
            decoder_input = decoder_input.cuda()

    print (' '.join(decoded_words))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Correction classifier')
    parser.add_argument('--train', '-t', required=True, type=str, help='file to train')
    parser.add_argument('--test', type=str, help='file to test')
    parser.add_argument('--load_en', type=str, help='encoder model to resume')
    parser.add_argument('--load_de', type=str, help='decoder model to resume')
    parser.add_argument('--max_length', type=int, default=25, help='max word length')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--n_layers', type=int, default=2, help="number of layers of GRU")
    parser.add_argument('--hidden_size', type=int, default=300, help="hidden size")
    global args
    args = parser.parse_args()
    return args    

def setup():
    cudnn.benchmark = True
    test_dataset = MyDataset(args.train, filter_pair=True, max_length = 22, min_length = 3, max_word_length=args.max_length)
    voc_size = test_dataset.n_words
    
    encoder = C2WEncoderRNN(args.hidden_size, args.n_layers, dropout=args.dropout)
    decoder = BahdanauAttnDecoderRNN(voc_size, args.hidden_size, args.n_layers, dropout=args.dropout)
    print(encoder)
    print(decoder)
    print ("vocab size", voc_size)
    encoder.cuda()
    decoder.cuda()    

    if args.load_en and args.load_de:
        state_en = torch.load(args.load_en)
        state_de = torch.load(args.load_de)
        encoder.load_state_dict(state_en)
        decoder.load_state_dict(state_de)
        print('Loading parameters from {} {}'.format(args.load_en, args.load_de))

    return encoder, decoder, test_dataset


if __name__ == "__main__":
    try:
        args = parse_arguments()
        encoder, decoder, dataset = setup()
        while True:
            sent = input('Enter a sent: [no punc, cor at last]; "quit" to quit\n')
            if sent == "quit":
                break
            eval_randomly(sent.strip(), dataset, encoder, decoder)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
