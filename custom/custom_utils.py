import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torchvision import datasets,transforms
from functools import partial
import torch.distributions as tdist

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)

def performance_record_plot(args,fig_save_dir,performance_record):
    x = np.arange(len(performance_record)) + 1
    plt.plot(x, performance_record)
    if 'ADD' in args.task:
        plt.ylim(0.0, 0.20)
    plt.savefig(fig_save_dir + 'performance_record.png')
    # plt.show()

def get_adding_batch(args):
    """Generate the adding problem dataset"""
    # Build the first sequence

    add_values = torch.rand(
        args.maxlen, args.batch_size, requires_grad=False
    )

    # Build the second sequence with one 1 in each half and 0s otherwise
    add_indices = torch.zeros_like(add_values)
    half = int(args.maxlen / 2)
    for i in range(args.batch_size):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, args.maxlen)
        add_indices[first_half, i] = 1
        add_indices[second_half, i] = 1

    # Zip the values and indices in a third dimension:
    # inputs has the shape (time_steps, batch_size, 2)
    inputs = torch.stack((add_values, add_indices), dim=-1)
    targets = torch.mul(add_values, add_indices).sum(dim=0)
    return inputs, targets

def create_copy_dataset(size, T):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_copying_sequence(T)
        sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y

def generate_copying_sequence(T):
    tensor = torch.FloatTensor
    items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
    x = []
    y = []

    ind = np.random.randint(8, size=10)
    for i in range(10):
        x.append([items[ind[i]]])
    for i in range(T - 1):
        x.append([items[8]])
    x.append([items[9]])
    for i in range(10):
        x.append([items[8]])

    for i in range(T + 10):
        y.append([items[8]])
    for i in range(10):
        y.append([items[ind[i]]])

    x = np.array(x)
    y = np.array(y)

    return tensor([x]), torch.LongTensor([y])

def create_denoise_data(T, n_data, n_sequence):
    seq = np.random.randint(1, high=10, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, T + n_sequence - 1))

    for i in range(n_data):
        ind = np.random.choice(T + n_sequence - 1, n_sequence)
        ind.sort()
        zeros1[i][ind] = seq[i]

    zeros2 = np.zeros((n_data, T + n_sequence))
    marker = 10 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros2, seq), axis=1).astype('int64')

    return x, y

def to_var(x, on_cpu=False, gpu_id=None, async=False):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id, async)
    return x

def get_ifc_tensor(args,row_size):
    if args.model == "NLSTM":
        i_arr = to_var(torch.ones(row_size, args.hidden_size, 1))
        f_arr = to_var(torch.ones(row_size, args.hidden_size, 1))
        c_arr = to_var(torch.FloatTensor([]))
        c_dict = dict()
    else:
        i_arr,f_arr,c_arr,c_dict = None,None,None,None

    return i_arr,f_arr,c_arr,c_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def transform_flatten(tensor):
    return tensor.view(-1, 1).contiguous()

def transform_permute(tensor, perm):
    return tensor.index_select(0, perm)

def get_MNIST_dataset(args):
    if args.task == 'pMNIST':
        perm = torch.randperm(784)
    else:
        perm = torch.arange(0, 784).long()

    train_dataset = datasets.MNIST(
        root=args.data_dir, train=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                      transform_flatten,
                                      partial(transform_permute, perm=perm)]),
        download=True)

    test_dataset = datasets.MNIST(
        root=args.data_dir, train=False,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                      transform_flatten,
                                      partial(transform_permute, perm=perm)]),
        download=True)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000))
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000, 60000))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,sampler=train_sampler,num_workers=2)
    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,sampler=valid_sampler,num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return train_loader,valid_loader,test_loader


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)