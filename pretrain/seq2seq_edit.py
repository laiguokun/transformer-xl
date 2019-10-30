import random
import numpy as np
def get_ops_num(n_ops, add_ratio, del_ratio, rep_ratio):
  add_r = random.random() * add_ratio
  del_r = random.random() * del_ratio
  rep_r = random.random() * rep_ratio
  sum_r = add_r + del_r + rep_r
  n_add_ops = int(n_ops * add_r / sum_r)
  n_del_ops = int(n_ops * del_r / sum_r)
  n_rep_ops = n_ops - n_add_ops - n_del_ops
  return n_add_ops, n_del_ops, n_rep_ops

def check_idx(seq_len, rep_idx, del_idx, add_idx):
  rep_mask = np.zeros(seq_len, dtype=bool)
  rep_mask[rep_idx] = True
  del_mask = np.zeros(seq_len, dtype=bool)
  del_mask[del_idx] = True
  # remove (rep, del) equal to del
  if np.logical_and(rep_mask, del_mask).sum() > 0:
    return False
  add_mask = np.zeros(seq_len, dtype=bool)
  add_mask = np.insert(add_mask, add_idx, True)
  del_mask = np.insert(del_mask, add_idx, False)
  rep_mask = np.insert(rep_mask, add_idx, False)
  # remove (add, rep) equal to (rep, add)
  rep_left_shift = rep_mask[1:]
  add_right_shift = add_mask[:-1]
  if np.logical_and(rep_left_shift, add_right_shift).sum() > 0:
    return False
  # remove (del, add) equal to rep
  add_left_shift = add_mask[1:]
  del_right_shift = del_mask[:-1]
  if np.logical_and(add_left_shift, del_right_shift).sum() > 0:
    return False
  # remove (add, del) equal to none
  add_right_shift = add_mask[:-1]
  del_left_shift = del_mask[1:]
  if np.logical_and(add_right_shift, del_left_shift).sum() > 0:
    return False
  return True

def add_edit_noise_to_sample(
  x, 
  mask_idx, 
  add_ph_idx,
  add_ratio=1., 
  del_ratio=1., 
  rep_ratio=1.,
  ops_ratio=0.15,
  has_eos=False):
  '''
    input:
      x: the sample, numpy vector with shape (seqlen, )
      mask_idx: mask token id
      add_ph_idx: insert ops placeholder id
      add_ratio: ratio of insert ops among editing ops
      del_ratio: ratio of delele ops among editing ops
      rep_ratio: ratio of replace ops among editing ops
      ops_ratio: decide the number of editing ops: n_ops = seqlen * ops_ratio
      has_eos: assume the sentence don't have <eos>. The output don't have <eos>
    output:
      y_encoder: the input sample for encoder
      label_encoder: the ops label for the encoder
      y_decoder: the input sample for decoder
      label_decoder: the ops label for the decoder
  '''
  if has_eos:
    x = x[:-1]
  seq_len = x.shape[0]
  n_ops = int(seq_len * ops_ratio)
  n_add_ops, n_del_ops, n_rep_ops = get_ops_num(
    n_ops, add_ratio, del_ratio, rep_ratio)
  
  label_decoder = np.zeros(seq_len, dtype=int)
  label_encoder = np.zeros(seq_len, dtype=int)
  x_decoder = x
  x_encoder = np.copy(x)

  rep_idx = np.random.choice(x.shape[0], size=n_rep_ops, replace=False)
  del_idx = np.random.choice(x.shape[0], size=n_del_ops, replace=False)
  add_idx = np.random.choice(x.shape[0], size=n_add_ops)
  #rep_idx = [9]
  #del_idx = [1]
  #add_idx = [2,2]
  #print(rep_idx, del_idx, add_idx)
  while not check_idx(seq_len, rep_idx, del_idx, add_idx):
    rep_idx = np.random.choice(x.shape[0], size=n_rep_ops, replace=False)
    del_idx = np.random.choice(x.shape[0], size=n_del_ops, replace=False)
    add_idx = np.random.choice(x.shape[0], size=n_add_ops)
    #print('resample', rep_idx, del_idx, add_idx)    
  # the editing ops order is (add, del, rep)
  # in the label vector (0:orig, 1:add, 2:del, 3:rep)
  # rep ops
  rep_mask = np.zeros(seq_len, dtype=bool)
  rep_mask[rep_idx] = True
  x_encoder[rep_mask] = mask_idx
  label_encoder[rep_mask] = 3
  label_decoder[rep_mask] = 3
  # add ops
  x_decoder = np.insert(x_decoder, add_idx, add_ph_idx)
  x_encoder = np.insert(x_encoder, add_idx, mask_idx)
  label_decoder = np.insert(label_decoder, add_idx, 1)
  label_encoder = np.insert(label_encoder, add_idx, 1)
  # del ops
  del_mask = np.ones(seq_len, dtype=bool)
  del_mask[del_idx] = False
  del_mask = np.insert(del_mask, add_idx, True)
  x_encoder = x_encoder[del_mask]
  label_encoder = label_encoder[del_mask]
  label_decoder[np.logical_not(del_mask)] = 2 
  
  return x_encoder, label_encoder, x_decoder, label_decoder

if __name__ == "__main__":
  x = np.arange(10)
  mask_idx = -1
  add_ph_idx = -2
  while True:
    x1, l1, x2, l2 = add_edit_noise_to_sample(x, -1, -2, ops_ratio=0.5)
    print('encoder info')
    print(x1)
    print(l1)
    print('decoder info')
    print(x2)
    print(l2)
    input()
