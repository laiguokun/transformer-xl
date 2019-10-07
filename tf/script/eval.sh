#!/bin/bash

# data
seq_len=512
mem_len=512
batch_size=8
num_host=1
num_core=1

record_dir=gs://language_model/wt103
# model_dir=gs://language_model/exp/wt103-512-3
model_dir=gs://language_model/exp/wt103-debug

# model
n_layer=1
d_model=1024
d_embed=1024
n_head=16
d_head=64
d_inner=4096
dropout=0.5
dropatt=0.1
dropact=0.1

ff_activation=gelu
use_lm_proj=False

python eval.py \
    --record_dir=${record_dir} \
    --model_dir=${model_dir} \
    --n_layer=${n_layer} \
    --d_model=${d_model} \
    --d_embed=${d_embed} \
    --n_head=${n_head} \
    --d_head=${d_head} \
    --d_inner=${d_inner} \
    --dropout=${dropout} \
    --dropatt=${dropatt} \
    --dropact=${dropact} \
    --ff_activation=${ff_activation} \
    --use_lm_proj=${use_lm_proj} \
    --optimizer=${optimizer} \
    --warmup_steps=20000 \
    --train_steps=100000 \
    --seq_len=${seq_len} \
    --mem_len=${mem_len} \
    --num_passes=10 \
    --lower_case=False \
    --eval_batch_size=${batch_size} \
    --num_hosts=${num_host} \
    --num_core_per_host=${num_core} \
    --iterations=5000 \
    --save_steps=10000 \
    --use_tpu=True \
    --use_bfloat16=True \
    --float32_softmax=True \
    --tpu=mid_1 \
    --tpu_zone=europe-west4-a \
    --tokenizer_type=sent_piece \
    --tokenizer_paths=../data/wikitext-103-sp/wt103.model \
    ${@}
