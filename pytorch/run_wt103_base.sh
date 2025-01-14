#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/wikitext-103-sp/ \
        --dataset wt103 \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 200 \
        --mem_len 200 \
        --eval_tgt_len 200 \
        --batch_size 68 \
        --multi_gpu \
        --gpu0_bsz 12 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --renormalize \
        --cuda \
        --data ../data/wikitext-103-sp/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        ${@:2}
elif [[ $1 == 'get_hidden' ]]; then
    echo 'get hidden rep'
    python get_hidden_rep.py \
        --cuda \
        --data ../data/wikitext-103-sp/ \
        --dataset wt103 \
        --tgt_len 384 \
        --mem_len 384 \
        --batch_size 64 \
        --clamp_len 400 \
        --same_length \
        ${@:2}
elif [[ $1 == 'eval_knn' ]]; then
    echo 'eval knn'
    python evalwknn.py \
        --cuda \
        --data ../data/wikitext-103-sp/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --batch_size 10 \
        --clamp_len 400 \
        --same_length \
        --split test \
        --no_log \
        ${@:2}
else
    echo 'unknown argment 1'
fi
