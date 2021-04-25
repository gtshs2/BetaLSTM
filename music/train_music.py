import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
# sys.path.append("../../")
from utils import data_generator
import numpy as np
import time
import datetime
import os
from os.path import dirname
import logging
import random

curpath = dirname(__file__)
parpath = os.path.dirname(curpath)
if parpath == "":
    parpath = ".."
tarpath = parpath + "/custom"
print(tarpath)
sys.path.append(tarpath)
import custom_rnn
import custom_model
from custom_utils import repackage_hidden,kl_anneal_function,get_MNIST_dataset

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--model', type=str, default='lstm',
                       choices='[nnlstm,lstm,flstm,noisin,g2lstm,ibetalstm,bbetalstm,gbetalstm]')
parser.add_argument('--max_epoch', type=int, default=2,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument("--path", type=str, default="data/", help="path to corpus directory")
parser.add_argument('--dataset', type=str, default='Piano',
                    help='the dataset to run (default: Nott), candidates : JSB, Muse, Nott, Piano')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument("--cv", type=int, default=0)
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--experiment', type=str, default='0507',
                    help='experiment code')
parser.add_argument('--rnn_d', type=int, default=200,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--model_size', type=str, default='small',
                       choices='[small,medium]')
''' '''
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument("--depth", type=int, default=1)
''' SAVE '''
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_dir', default="results/", help='')
parser.add_argument('--log_dir', default="logs/")
parser.add_argument('--random_seed', default=1000, type=int)

''' GBETALSTM '''
parser.add_argument('--use_kl_anneal', action='store_true')
parser.add_argument('--kl_lambda', type=float, default=1.0)
parser.add_argument('--kl_gamma_prior', type=float, default=0.5)
parser.add_argument('--kl_k', type=float, default=0.20)
parser.add_argument('--kl_x0', type=float, default=50)

''' NOISIN '''
parser.add_argument('--noisin_noise_type', type=str, default='bernoulli', choices=['gamma', 'bernoulli'])
parser.add_argument('--noisin_noise_parama', type=float, default=0.41)
parser.add_argument('--noisin_noise_paramb', type=float, default=0.41)

''' For G2LSTM '''
parser.add_argument('--gumbel_noise_p', type=float, default=0.2,
                       help='Gmubel_noise_p in Gumbel gate')
parser.add_argument('--gumbel_noise_t', type=float, default=1.0,
                       help='Gmubel_noise_t in Gumbel gate')
parser.add_argument('--gumbel_noise_type', type=str, default='new_U_B',
                       help='Gmubel_noise_type in Gumbel gate (new_U_B, new_U, no_new)')
parser.add_argument('--divide_temp', type=float, default=0.9,
                       help='Temperature in LSTM gates')

args = parser.parse_args()

if args.dataset in ["Nott","Muse","Piano","JSB"]:
    if "flstm" in args.model.lower():
        args.rnn_d = 232
    elif "ibeta" in args.model.lower():
        args.rnn_d = 161
    elif "bbeta" in args.model.lower():
        args.rnn_d = 177
    elif "gbeta" in args.model.lower():
        args.rnn_d = 148
    else:
        args.rnn_d = 200
else:
    raise ValueError("No dataset")

args.cuda = torch.cuda.is_available()
np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

# args.save_dir = args.save_dir + args.dataset + "/" + args.model + "_" + str(args.model_size) + "_"+"/"
args.save_dir = args.save_dir + "{}_{}_{}_{}_{}_{}_{}_{}/".format(str(args.dataset),str(args.cv),str(args.random_seed),
                                                              str(args.model),str(args.model_size),
                                                              str(args.kl_lambda),str(args.kl_gamma_prior),
                                                              str(args.use_kl_anneal))
args.model_save_dir = args.save_dir + "model/"
args.fig_save_dir = args.save_dir + "figs/"
os.makedirs(args.model_save_dir) if not os.path.exists(args.model_save_dir) else 1
os.makedirs(args.fig_save_dir) if not os.path.exists(args.fig_save_dir) else 1

args.task = args.dataset
args.data_dir = args.path + args.task + "/"

os.makedirs(args.log_dir) if not os.path.exists(args.log_dir) else 1
PATH_LOG = args.log_dir + "logger_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7},txt".\
    format(str(args.dataset),str(args.cv), str(args.model),args.model_size,str(args.kl_lambda),
           str(args.kl_gamma_prior),str(args.use_kl_anneal),args.random_seed)
logger = logging.getLogger('Result_log')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(PATH_LOG)
logger.addHandler(file_handler)

logger.info("==" * 10)
print("==" * 10)
for param in vars(args).keys():
    s = '--{0} : {1}'.format(param, vars(args)[param])
    logger.info(s)
    print(s)
logger.info("==" * 10)
print("==" * 10)

class Model(nn.Module):
    def __init__(self, args,inp_d=None,nclasses=None):
        super(Model, self).__init__()
        args.dropouth = args.dropout
        args.dropouti = args.dropoute = args.wdrop = 0.0
        # self.drop = nn.Dropout(args.dropout)
        self.encoder = custom_model.RNNModel(args=args,rnn_type=args.model,ninp=inp_d,nhid=args.rnn_d,
                                                 nlayers=args.depth,dropout = args.dropout,tie_weights=False)
        d_out = args.rnn_d
        self.out = nn.Linear(d_out, nclasses)
        self.sig = nn.Sigmoid()

    def forward(self, input,init_hidden=None):
        output, hidden,gate_value,total_kl_loss = self.encoder(input,init_hidden,len_input=None)
        # output : [maxlen,batch_size,hidden_dim]
        # output = output[-1] # real_last_output
        #output = output[-1,:,:]

        # output = self.drop(output)
        output = output.reshape(output.size(0) * output.size(1), output.size(2))
        return self.sig(self.out(output)),gate_value,total_kl_loss

input_size = 88
X_train, X_valid, X_test = data_generator(args.dataset)

dropout = args.dropout

#model = TCN(input_size, input_size, n_channels, kernel_size, dropout=args.dropout)
model = Model(args,inp_d=input_size, nclasses=input_size).cuda()

num_total_parameters = sum(p.numel() for p in model.parameters())
num_grad_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
s = "Total Number of Parameters : {}/{}".format(num_total_parameters, num_grad_parameters)
logger.info(s)
print(s)

# model_spec = str(args.data)+'_'+str(args.dropout)+'_'+str(args.clip)+'_'+str(args.depth)+'_'+str(args.lr)+'_'+str(args.rnn_d)+'_'+str(args.optim)+'_'+str(args.experiment)+'.txt'
# if not os.path.exists(model_spec):
#     f = open(model_spec, 'w')
#     f.write('Results for model :: \n')
#     f.close()

def detach(states):
    return [state.detach() for state in states]

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
if args.optim =='SGD':
    print('optimizer is SGD')
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    print('optimizer is Adam')
    optimizer = optim.Adam(model.parameters(), lr=lr)

def evaluate(X_data, name='Final Best Test'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")

    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            hidden = model.encoder.init_hidden(1)
            hidden = repackage_hidden(hidden)
            output, gate_value,total_kl_loss = model(x.unsqueeze(1),hidden)
            output = output.squeeze(0)
            #y = y.long()
            loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                                torch.matmul((1-y), torch.log(1-output).float().t()))
            total_loss += loss.item()
            count += output.size(0)

        eval_loss = total_loss / count
        s = name + " loss: {:.5f}".format(eval_loss)
        print(s)
        logger.info(s)
        return eval_loss


def train(ep):
    model.train()
    all_loss = 0
    all_y_loss = 0
    all_kl_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    #
    # states = (torch.zeros(args.levels, 1, args.nhid).to(device),
    #           torch.zeros(args.levels, 1, args.nhid).to(device))


    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()  # x : [maxlen,inp_dim]

        x = x.unsqueeze(1)
        len_x = x.size()[0]
        model.zero_grad()
        hidden = model.encoder.init_hidden(1)
        hidden = repackage_hidden(hidden)
        output, gate_value,total_kl_loss = model(x,hidden) # [maxlen,1,inp_dim]
        output = output.squeeze(0)

        # print("=====")
        # print(total_kl_loss.size())

        num_sent = x.size()[1]
        if args.model == "gbetalstm":
            kl_loss = torch.tensor(0.0).cuda()
            for itr in range(num_sent):
                kl_loss += total_kl_loss[:, :, itr].sum()
            # kl_loss = kl_loss / (len_x * num_sent)

            kl_loss = args.kl_lambda * kl_loss
            y_loss = -torch.trace(torch.matmul(y, torch.log(output + (1e-8)).float().t()) +
                                  torch.matmul((1 - y), torch.log(1 - output + (1e-8)).float().t()))
            loss = y_loss + kl_loss
        else:
            kl_loss = torch.tensor(0.0)
            y_loss = -torch.trace(torch.matmul(y, torch.log(output + (1e-8)).float().t()) +
                                  torch.matmul((1 - y), torch.log(1 - output + (1e-8)).float().t()))
            loss = y_loss

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()

        all_loss += loss.item()
        all_y_loss += y_loss.item()
        all_kl_loss += kl_loss.item()
        count += output.size(0)
        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = all_loss / count
            cur_y_loss = all_y_loss / count
            cur_kl_loss = all_kl_loss / count
            s = "Epoch {:2d} | lr {:.5f} | Total loss {:.5f} | y loss {:5f} | kl loss {:5f}".format(ep, lr, cur_loss,cur_y_loss,cur_kl_loss)
            print(s)
            logger.info(s)
            all_loss = 0.0
            all_y_loss = 0.0
            all_kl_loss = 0.0
            count = 0

if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []

    model_name = args.model_save_dir + "bestmodel.pt"
    #model_name = "poly_music_{0}.pt".format(args.dataset)

    for ep in range(1, args.max_epoch+1):
        epoch_start = time.time()
        train(ep)
        vloss = evaluate(X_valid, name='Validation')
        tloss = evaluate(X_test, name='Test')
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 1.2
            # lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)
        epoch_end = time.time()
        s0 = "One Epoch Training Duration : {}".format(str(datetime.timedelta(seconds=(epoch_end-epoch_start))))
        s1 = "-" * 50
        s = '\n'.join(string for string in [s0, s1])
        print(s)
        logger.info(s)

    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(X_test)
    s = 'Whole Training Duration : ', str(datetime.timedelta(seconds=(time.time() - epoch_start)))
    print(s)
    logger.info(s)

    s = "{} {} {} {} {} {} {} {} / valid: {:.6f} / Test: {:.6f}".\
        format(str(args.dataset),str(args.cv),str(args.random_seed),str(args.model),str(args.model_size),
               str(args.kl_lambda),str(args.kl_gamma_prior),str(args.use_kl_anneal)
               ,best_vloss,tloss)
    import logging
    PATH_LOG = args.log_dir + "overall_logger_{0},txt".format(str(args.dataset))
    global_logger = logging.getLogger('Result_log')
    global_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(PATH_LOG)
    global_logger.addHandler(file_handler)
    global_logger.info(s)