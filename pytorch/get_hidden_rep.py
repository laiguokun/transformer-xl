# coding: utf-8
import argparse
import time
import math
import os, sys

import torch
import numpy as np
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='train',
                    choices=['train'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
parser.add_argument('--save_dir', type=str, default='./tmp/')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    start_time = time.time()
    cnt = 0
    hidden_list = []
    target_list = []
    segment = eval_iter.n_batch // 10
    print("total batch number {}, save per batch {}".format(eval_iter.n_batch, segment))
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = model.get_hidden_rep(data, target, *mems)
            hidden, mems = ret[0], ret[1:]
            hidden = hidden.view(-1, hidden.size(-1))
            target = target.view(-1, target.size(-1))
            hidden_list.append(hidden.cpu().numpy().astype('float16'))
            target_list.append(target.cpu().numpy())
            cnt += 1
            if cnt % segment == 0:
                hidden_list = np.concatenate(hidden_list, axis=0)
                target_list = np.concatenate(target_list, axis=0)
                hidden_f = os.path.join(args.save_dir, 'hidden_{}.npy'.format(cnt//segment))
                target_f = os.path.join(args.save_dir, 'target_{}.npy'.format(cnt//segment))
                np.save(hidden_f, hidden_list)
                np.save(target_f, target_list)
                hidden_list = []
                target_list = []
        total_time = time.time() - start_time
    if len(hidden_list)!=0:
        hidden_list = np.concatenate(hidden_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        hidden_f = os.path.join(args.save_dir, 'hidden_{}.npy'.format(cnt//segment + 1))
        target_f = os.path.join(args.save_dir, 'target_{}.npy'.format(cnt//segment + 1))
        np.save(hidden_f, hidden_list)
        np.save(target_f, target_list)
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))
    return 0.

# Run on test data.
train_loss = evaluate(tr_iter)

logging('=' * 100)
