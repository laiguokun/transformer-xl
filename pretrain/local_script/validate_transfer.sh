#!/bin/bash

# model
n_layer=14
d_model=1024
d_embed=1024
n_head=16
d_head=64
d_inner=4096
dropout=0.1
dropatt=0.1
dropact=0.0

python validate_transfer.py \
    --n_layer=${n_layer} \
    --d_model=${d_model} \
    --d_embed=${d_embed} \
    --n_head=${n_head} \
    --d_head=${d_head} \
    --d_inner=${d_inner} \
    --dropout=${dropout} \
    --dropatt=${dropatt} \
    --tokenizer_type=whitespace \
    --tokenizer_path=${HOME}/data/mt/vocab.txt \
    --init_checkpoint=${HOME}/data/mt/ckpt/model_500K/model.ckpt-0 \
    ${@}
