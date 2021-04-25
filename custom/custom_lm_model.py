import torch
import torch.nn as nn

from custom_dropout import LockedDropout,embedded_dropout,WeightDrop
import custom_rnn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    #def __init__(self, args,rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
    def __init__(self, args, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.dropouti = args.dropouti
        self.dropouth = args.dropouth
        self.dropoute = args.dropoute
        self.dropout = args.dropout
        self.wdrop = args.wdrop
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(self.dropouti)
        self.hdrop = nn.Dropout(self.dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        #assert rnn_type in ['nnlstm'], 'RNN type is not supported'
        if rnn_type == 'nnlstm':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if self.wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=self.wdrop) for rnn in self.rnns]
        elif rnn_type == 'lstm':
            self.rnns = [custom_rnn.LSTM(args,custom_rnn.LSTMCell,
                                      ninp if l == 0 else nhid,
                                      nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                      1,
                                      dropout=0,
                                      wdrop=self.wdrop) for l in range(nlayers)]
        elif rnn_type == 'flstm':
            self.rnns = [custom_rnn.LSTM(args,custom_rnn.FLSTMCell,
                                      ninp if l == 0 else nhid,
                                      nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                      1,
                                      dropout=0,
                                      wdrop=self.wdrop) for l in range(nlayers)]
        elif rnn_type == 'noisin':
            self.rnns = [custom_rnn.LSTM(args,custom_rnn.NoisinCell,
                                      ninp if l == 0 else nhid,
                                      nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                      1,
                                      dropout=0,
                                      wdrop=self.wdrop,
                                         noisin_noise_type=args.noisin_noise_type,
                                         noisin_noise_parama=args.noisin_noise_parama,
                                         noisin_noise_paramb=args.noisin_noise_paramb) for l in range(nlayers)]
        elif rnn_type == 'g2lstm':
            self.rnns = [custom_rnn.LSTM(args,custom_rnn.G2LSTMCell,
                                      ninp if l == 0 else nhid,
                                      nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                      1,
                                      dropout=0,
                                      wdrop=self.wdrop,
                                      gumbel_noise_p=args.gumbel_noise_p,
                                      gumbel_noise_t=args.gumbel_noise_t,
                                      gumbel_noise_type=args.gumbel_noise_type,
                                      divide_temp=args.divide_temp) for l in range(nlayers)]
        elif rnn_type == 'ibetalstm':
            self.rnns = [custom_rnn.LSTM(args,custom_rnn.iBetaLSTMCell,
                                      ninp if l == 0 else nhid,
                                      nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                      1,
                                      dropout=0,
                                      wdrop=self.wdrop,
                                         eps=args.eps) for l in range(nlayers)]
        elif rnn_type == 'bbetalstm':
            self.rnns = [custom_rnn.LSTM(args,custom_rnn.bBetaLSTMCell,
                                      ninp if l == 0 else nhid,
                                      nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                      1,
                                      dropout=0,
                                      wdrop=self.wdrop,
                                         eps=args.eps) for l in range(nlayers)]
        elif rnn_type == 'gbetalstm':
            self.rnns = [custom_rnn.LSTM(args,custom_rnn.gBetaLSTMCell,
                                      ninp if l == 0 else nhid,
                                      nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                      1,
                                      dropout=0,
                                      wdrop=self.wdrop,
                                         eps=args.eps,
                                         kl_gamma_prior=args.kl_gamma_prior) for l in range(nlayers)]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        gate_value = ()
        total_kl_loss = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h,gate_value,kl_loss = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            total_kl_loss.append(kl_loss)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        try: # for gbeta case
            total_kl_loss = torch.stack(total_kl_loss, 0)
        except:
            total_kl_loss = torch.tensor(0.0)

        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden,gate_value,total_kl_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'nnlstm':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        else:
            return [[(weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())]
                     for l in range(self.nlayers)]