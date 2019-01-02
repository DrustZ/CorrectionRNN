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
import socket
hostname = socket.gethostname()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import torchvision
from PIL import Image
import visdom
vis = visdom.Visdom()

attn_images = []

def show_plot_visdom():
    global attn_images
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    img = torchvision.transforms.ToTensor()(Image.open(buf))
    
    if len(attn_images) > 5:
        attn_images = attn_images[1:]+[img]
    else:
        attn_images += [img]
    vis.image(torch.cat(attn_images, dim=1), win=attn_win, opts={'title': attn_win})

def show_attention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    
#     print (attentions.size())
    # Set up axes
    ax.set_xticklabels([''] + input_words + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()

def evaluate_and_show_attention(input_words, target_sentence, output_words, attentions):
    output_sentence = ' '.join(output_words)
    input_sentence = ' '.join(input_words)

    print('>', input_sentence)
    if target_sentence is not None:
        print('=', ' '.join(target_sentence))
    print('<', output_sentence)
    
    show_attention(input_words, output_words, attentions)
    
    # Show input, target, output text in visdom
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    vis.text(text, win=win, opts={'title': win})

PAD_TOKEN = 0
SOS_token = 1
EOS_token = 2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Correction classifier')
    parser.add_argument('--train', '-t', required=True, type=str, help='file to train')
    parser.add_argument('--test', type=str, help='file to test')
    parser.add_argument('--load_en', type=str, help='encoder model to resume')
    parser.add_argument('--load_de', type=str, help='decoder model to resume')
    parser.add_argument('--max_length', type=int, default=20, help='max word length')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--elr', type=float, default=1e-4, help='encoder learning rate')
    parser.add_argument('--dlr', type=float, default=5e-4, help='decoder learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--clip', type=float, default=False, help='grad clipping')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--step', type=int, default=100, help='decay lr every * epochs')
    parser.add_argument('--test_freq', type=int, default=5, help='test every * epochs')
    parser.add_argument('--n_layers', type=int, default=2, help="number of layers of GRU")
    parser.add_argument('--hidden_size', type=int, default=300, help="hidden size")
    parser.add_argument('--teacher_forcing_ratio', '-teacher', type=float, default=0.5, help="teacher_forcing_ratio")    
    global args
    args = parser.parse_args()
    return args    

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

def eval_randomly(test_dataset, encoder, decoder, max_length=20): #changed to 20 max
    pair = test_dataset.pairs[random.randint(0, len(test_dataset)-1)]
    input_seq = pair[0]
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

        target_sentence = pair[1]
        evaluate_and_show_attention(input_seq, target_sentence, decoded_words, decoder_attentions)
    

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

def train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio):
    encoder.train()
    decoder.train()
    total_loss = 0
    accuracy = 0
    cnt = 0

    for i, (src, len_src, trg, len_trg) in enumerate(tqdm(train_data, leave=False)):
        src = src.cuda() #T B V
        trg = trg.cuda() #T B
        batch_size = src.shape[1]
        # Zero gradients of both optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

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
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = decoder_output.max(1)[1]
            top1s[t] = top1.cpu()
            decoder_input = (trg[t] if is_teacher else top1).cuda()

        loss = F.nll_loss(all_decoder_outputs.view(-1, decoder.output_size),
                               trg.contiguous().view(-1),
                               ignore_index=PAD_TOKEN)
        # loss = masked_cross_entropy(
        #             all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        #             trg.transpose(0, 1).contiguous(), # -> batch x seq
        #             len_trg
        #         )
        loss.backward()

        accuracy += correct(top1s, trg.cpu(), len_trg)

        cnt += int(src.shape[1])
        total_loss += float(loss.item())
        if args.clip:
            # Clip gradient norms
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

        # Update parameters with optimizers
        encoder_optimizer.step()
        decoder_optimizer.step()
    
    return total_loss / cnt, accuracy / cnt

def main(args):
    cudnn.benchmark = True
    train_dataset = MyDataset(args.train, filter_pair=True, max_length = args.max_length, min_length = 3, max_word_length=args.max_length)
    voc_size = train_dataset.n_words
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                            shuffle=True, num_workers=2, collate_fn=collate_fn())

    test_dataset = MyDataset(args.train, filter_pair=True, max_length = args.max_length, min_length = 3, max_word_length=args.max_length, train=False)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                            shuffle=True, num_workers=2, collate_fn=collate_fn())

    encoder = C2WEncoderRNN(args.hidden_size, args.n_layers, dropout=args.dropout)
    decoder = BahdanauAttnDecoderRNN(voc_size, args.hidden_size, args.n_layers, dropout=args.dropout)
    print(encoder)
    print(decoder)
    print ("vocab size", voc_size)
    encoder.cuda()
    decoder.cuda()

    if args.load_en:
        state_en = torch.load(args.load_en)
        encoder.load_state_dict(state_en)
        print('Loading parameters from {}'.format(args.load_en))

    if args.load_de:
        state_de = torch.load(args.load_de)
        decoder.load_state_dict(state_de)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.elr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.dlr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, verbose=True)

    best_acc = 0
    for e in range(1, args.epochs+1):
        loss, acc = train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer, args.teacher_forcing_ratio)
        print('Epoch {}:\tavg.loss={:.4f}\tavg.acc={:.4f}'.format(e, loss, acc))

        torch.cuda.empty_cache()

        if args.test and args.test_freq and e % args.test_freq == 0:
            eval_randomly(test_dataset, encoder, decoder)
            loss, acc = evaluate(test_data, encoder, decoder)
            torch.cuda.empty_cache()
            ind = ''
            if acc > best_acc:
                best_acc = acc
                ind = '*'
                torch.save(encoder.state_dict(), 'best_en_5out_amazon.pth')
                torch.save(decoder.state_dict(), 'best_de_5out_amazon.pth')
            print('----Validation:\tavg.loss={:.4f}\tavg.acc={:.4f}{}'.format(loss, acc, ind))
    print('Best Accuracy: {}'.format(best_acc))

if __name__ == "__main__":
    try:
        args = parse_arguments()
        main(args)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
