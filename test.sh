#!/bin/bash

cp  epic_kitchen/epic100_cp.jsonl  epic_kitchen/epic100.jsonl 
python inference_sthv2.py   --dist_number 1 \
  --codebook_size 8 \
  --laq_checkpoint ./laq_openx.pt \
  --divider 1 \
  --window_size 16 \
  --code_seq_len 256 \
  --layer 2 --input_file ./epic_kitchen/epic100.jsonl --unshuffled_jsonl ./epic_kitchen/epic100.jsonl

