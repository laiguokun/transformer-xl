#### Prerequisite

- Pytorch 0.4: `conda install pytorch torchvision -c pytorch`



#### Data Prepration

`bash getdata.sh`



#### Replicate the "bpc = 1.06" result on `enwik8` with a 12-layer Transformer-XL

- Make sure the machine have **4 GPUs**, each with **at least 11G memory**

- Training

  `bash run_enwik8.sh train --work_dir PATH_TO_WORK_DIR`

- Testing

  `bash run_enwik8.sh eval --work_dir PATH_TO_WORK_DIR`



#### Replicate the "PPL = 24.03" result on `wikitext-103` with Transformer-XL

- Make sure the machine have **4 GPUs**, each with **at least 11G memory**

- Training

  `bash run_wt103.sh train --work_dir PATH_TO_WORK_DIR`

- Testing

  `bash run_wt103.sh eval --work_dir PATH_TO_WORK_DIR`



#### Other options:

- `--batch_chunk`: this option allows one to trade speed for memory. For `batch_chunk > 1`, the program will split each training batch into `batch_chunk` sub-batches and perform forward and backward on each sub-batch sequentially, with the gradient accumulated and divided by `batch_chunk`. Hence, the memory usage will propertionally lower while the computation time will inversely higher. 
- `--div_val`: when using adaptive softmax and embedding, the embedding dimension is divided by `div_val` from bin $i$ to bin $i+1$. This saves both GPU memory and the parameter budget.
- `--fp16` and `--dynamic-loss-scale`: Run in pseudo-fp16 mode (fp16 storage fp32 math) with dynamic loss scaling. 
  - Note: to explore the `--fp16` option, please make sure the `apex` package is installed (https://github.com/NVIDIA/apex/).
- `--attn_type`: set `attn_type` to 2 to use standard Transformer without any recurrence.



#### Other datasets:

- `Text8` character-level language modeling: check out `run_text8.sh`
- `lm1b` word-level language modeling: check out `run_lm1b.sh`