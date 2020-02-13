import math
import torch
import random
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

class C2WEncoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0.5, num_char = 65, char_emb_dim=15):
        super(C2WEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = 300
        self.n_layers = n_layers
        self.dropout = dropout
        self.attn = Attn('concat', hidden_size)
        self.char_embed = nn.Embedding(num_char, char_emb_dim)
        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (75, 4), (75, 5)]
        self.convolutions = nn.ModuleList()
        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(nn.Conv1d(char_emb_dim, out_channel, kernel_size=filter_width, padding=0))
        
        self.batch_norm = nn.BatchNorm1d(self.embed_size)
        self.gru = nn.GRU(self.embed_size*2, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, x, input_lengths, hidden=None):
        '''
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B), n_word(W)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        # Input: Variable of Tensor with shape [batch, words, max_word_len+2]
        T, B = int(x.shape[0]), int(x.shape[1])
        #input = input.transpose(1, 2)
        x = x.view(-1, x.size()[2])
        # [batch*words, max_word_len+2]

        x = self.char_embed(x)
        # [batch*words, max_word_len+2, char_emb_dim]

        x = x.transpose(1, 2)
        # [batch*words, char_emb_dim, max_word_len+2]

        x = self.conv_layers(x)
        x = self.batch_norm(x)
        x = x.view(T, B, -1)
        #[T,B,E]
        
        # model 2: embedding concat
        #select corrections (corrections are the last ones)
        corrections = torch.cat([x[idx1-1, idx2].unsqueeze(0) for idx1, idx2 in zip(input_lengths, range(B))])
        corrections = corrections.unsqueeze(0)
        #corrections : [1, B, E]
        x = torch.cat((x, corrections.repeat(T, 1, 1)), 2) #[T,B,2E]
        input_lengths = [h-1 for h in input_lengths]
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        
        return outputs, hidden
    
    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            # x = self.conv_layers(x)
            out = torch.tanh(conv(x))
            # (batch_size, out_channel, max_word_len-width+1)
            out = torch.max(out, 2)[0]
            # (batch_size, out_channel)
            chosen_list.append(out)
        
        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T, H]
        attn_energies = self.score(H,encoder_outputs, max_len, this_batch_size) # compute attention score
        return torch.softmax(attn_energies, dim=1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs, maxlen, bsize):
        catH = torch.cat([hidden, encoder_outputs], 2) # [B*T, 2H]
        energy = torch.tanh(self.attn(catH)) # [B*T,2H]->[B*T,H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(bsize,1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, dropout=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = 300
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        # Define layers
        self.embedding = nn.Embedding(output_size, self.embed_size)
        self.dropout = nn.Dropout(dropout)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + self.embed_size, hidden_size, n_layers, dropout=dropout)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = torch.log_softmax(self.out(output), dim=1)
        # Return final output, hidden state
        return output, hidden, attn_weights

