import faiss
import numpy as np
import torch

def to_onehot(x, c):
    bsz = x.size(0)
    x_onehot = torch.FloatTensor(bsz, c)
    x_onehot.zero_()
    x_onehot.scatter_(1, x, 1)
    return x_onehot

def safe_exp(x):
    max_x, _ = torch.max(x, 1, keepdim = True)
    x = x - max_x
    return torch.exp(x)

def retrive_items(table, idx):
    return np.take(table, idx, axis=0)

def get_knnp(index, hidden, topk, target_table, vocab_size, hidden_table=None):
    device = hidden.device
    hidden = hidden.view(-1, hidden.size(-1)).cpu()
    seq_len = hidden.size(0)
    
    dot_product = False
    if dot_product:
        #augment the hidden if using dot product
        bsz = hidden.size(0)
        extracol = torch.zeros(bsz, 1, device=hidden.device)
        hidden_ = torch.cat([hidden, extracol], -1)
        D_, I = index.search(hidden_.numpy(), topk)
    else:
        D_, I = index.search(hidden.numpy(), topk)
    idx = retrive_items(target_table, I).squeeze(-1)
    idx = torch.LongTensor(idx).to(device)
    if (hidden_table is None):
        D = torch.FloatTensor(D_).to(device)
        D = D.sqrt()
    else:
        hidden_retrived = retrive_items(hidden_table, I)
        hidden_retrived = torch.FloatTensor(hidden_retrived).to(device)
        if dot_product:
            D = calc_dot(hidden.to(device), hidden_retrived)
            # how to define the dist
            D = -D 
            D = torch.sign(D) * D.abs().sqrt()
            D = D
        else:
            D = calc_L2(hidden.to(device), hidden_retrived)
    #print(D[0])
    D = safe_exp(-D)
    knnp = sum_over_topk(idx, D, vocab_size)

    knnp_sum = torch.sum(knnp, -1, keepdim=True)
    knnp = knnp/knnp_sum
    #print(knnp.sum(-1))
    #knnp = torch.zeros(knnp.size(), device=knnp.device)
    return knnp

def calc_L2(hidden, retrived):
    bsz = retrived.size(0)
    hidden = hidden.view(bsz, 1, -1)
    D = torch.pow(hidden - retrived, 2)
    D = D.sum(-1).sqrt()
    return D

def calc_dot(hidden, retrived):
    bsz = retrived.size(0)
    hidden = hidden.view(bsz, 1, -1)
    D = (hidden * retrived).sum(-1)
    return D
    
def sum_over_topk(idx, D, vocab_size):
    p = torch.zeros(D.size(0), vocab_size+1, device=D.device)
    fake_token = vocab_size
    # sort and get cumsum
    sorted_idx, indices = torch.sort(idx, 1)
    sorted_D = torch.gather(D, 1, indices)
    sorted_D = sorted_D.cumsum(1)
    mask = torch.zeros(D.size(), device=D.device).byte()
    mask[:, :-1] = sorted_idx[:, :-1] == sorted_idx[:, 1:]
    sorted_idx = sorted_idx.masked_fill(mask, fake_token) 
    sorted_D = sorted_D.masked_fill(mask, 0)
    # remove cumsum by sorting
    idx_ = torch.arange(D.size(1), device=D.device).unsqueeze(0)
    idx_ = idx_.expand(D.size(0), D.size(1)).masked_fill(mask, -1)

    _, indices = torch.sort(idx_, 1)
    sorted_idx = torch.gather(sorted_idx, 1, indices)

    sorted_D = torch.gather(sorted_D, 1, indices)
    sorted_D[:, 1:] = sorted_D[:, 1:] - sorted_D[:, :-1]
    p = p.scatter_(1, sorted_idx, sorted_D)
    return p[:, :-1]



def read_target_table(dataset):
    print('loading target table')
    if dataset == 'wt103':
        target = []
        for i in range(1, 11):
            fn = './tmp/target_{}.npy'.format(i)
            #error
            x = np.load(fn).reshape(-1, 1)
            target.append(x) 
        target = np.concatenate(target, axis = 0)
        #target = torch.LongTensor(target)
    else:
        assert false
    return target


def read_hidden_table(dataset):
    print('loading hidden table')
    if dataset == 'wt103':
        hidden = np.load('/home/laiguokun/ssd/tmp/hidden_cache.npy')
        #hidden = np.load('./tmp/test.npy')
        #hidden = torch.FloatTensor(hidden)
    else:
        assert false
    return hidden