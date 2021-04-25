import torch
import torch.nn as nn

from custom_dropout import LockedDropout,embedded_dropout,WeightDrop
import custom_rnn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    #def __init__(self, args,rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
    def __init__(self, args, rnn_type, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.args = args
        self.dropouti = args.dropouti
        self.dropouth = args.dropouth
        self.dropoute = args.dropoute
        self.dropout = args.dropout
        self.wdrop = args.wdrop
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(self.dropouti)
        self.hdrop = nn.Dropout(self.dropouth)
        self.drop = nn.Dropout(dropout)

        # print("=====")
        # print(nlayers,dropout,self.dropout,self.dropouti,self.dropouth,self.dropoute,self.wdrop)
        # print(tie_weights)

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

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.tie_weights = tie_weights


    def forward(self, input, hidden, len_input, return_h=False):
        # emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # emb = self.lockdrop(emb, self.dropouti)

        raw_output = input
        new_hidden = []
        raw_outputs = []
        outputs = []
        gate_value = ()
        total_kl_loss = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h, gate_value,kl_loss = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            total_kl_loss.append(kl_loss)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout)
        try: # for gbeta case
            total_kl_loss = torch.stack(total_kl_loss, 0)
        except:
            total_kl_loss = torch.tensor(0.0)
        return output, hidden,gate_value,total_kl_loss

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