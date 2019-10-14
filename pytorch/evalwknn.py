# coding: utf-8
import argparse
import time
import math
import os, sys

import torch
import faiss

from data_utils import get_lm_corpus, get_renormalize_vocab
from mem_transformer import MemTransformerLM, EvalModel
from utils.exp_utils import get_logger
from utils.knnp import get_knnp, read_target_table, read_hidden_table

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
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
parser.add_argument('--lamb', type=float, default=0.1,
                    help='lambda of the evaluation function')
parser.add_argument('--topk', type=int, default=256,
                    help='lambda of the evaluation function')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset, renormalize=False)
ntokens = len(corpus.vocab)

va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)

# Load the best saved model.

with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {} topk {} lamb {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len, args.topk, args.lamb))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

print('loading index')
index = faiss.read_index('/home/laiguokun/ssd/tmp/wt103.index')
#index = faiss.read_index('/home/laiguokun/ssd/tmp/wt103_dot.index')
#index = faiss.read_index('./faiss/test.index')
index.nprobe = 32
target_table = read_target_table(args.dataset)
hidden_table = read_hidden_table(args.dataset)
#hidden_table = None
vocab_size = len(corpus.vocab)
print('vocab size: {}'.format(vocab_size))
###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    start_time = time.time()
    cnt = 0
    #last_status = renormalize_vocab.build_initial_status(args.batch_size)
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = model.get_hidden_rep(data, target, *mems)
            hidden, mems = ret[0], ret[1:]
            if args.lamb > 0:
                knnp = get_knnp(index, hidden, args.topk, 
                    target_table, vocab_size, hidden_table)
                loss = model.crit_with_knn(hidden, target, knnp, args.lamb).mean()
            else:
                loss = model.crit(hidden.view(-1, hidden.size(-1)), target.view(-1)).mean()
            total_loss += seq_len * loss
            total_len += seq_len
            cnt += 1
            #if cnt > 10:
            #    break
        total_time = time.time() - start_time
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))
    return total_loss / total_len

# Run on test data.
if args.split == 'all':
    test_loss = evaluate(te_iter)
    valid_loss = evaluate(va_iter)
elif args.split == 'valid':
    valid_loss = evaluate(va_iter)
    test_loss = None
elif args.split == 'test':
    test_loss = evaluate(te_iter)
    valid_loss = None

def format_log(loss, split):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str

log_str = ''
if valid_loss is not None:
    log_str += format_log(valid_loss, 'valid')
if test_loss is not None:
    log_str += format_log(test_loss, 'test')

logging('=' * 100)
logging(log_str)
logging('=' * 100)
