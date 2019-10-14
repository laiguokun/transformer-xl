#!/bin/bash

split=valid
python create_lm_data.py \
    --seq_len=16384 \
    --bsz_per_host=8 \
    --lower_case=False \
    --split=${split} \
    --tokenizer_type=sent_piece \
    --tokenizer_paths=../data/wikitext-103-sp/wt103.model \
    --input_glob=../data/wikitext-103/${split}.txt \
    --save_dir=../data/wikitext-103/tfrecords \
    --num_task=1 \
    --task=0 \
    --add_eos=True \
    $@
