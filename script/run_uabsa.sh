#!/usr/bin/env bash

name=`basename "$0"`
name=${name%???} #stript .sh


python main.py --task uabsa \
            --name $name \
            --seed 42 \
            --dataset cross_domain \
            --model_name_or_path t5-base \
            --paradigm extraction-universal \
            --n_gpu 0 \
            --train_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 128 \
            --learning_rate 3e-4 \
            --num_train_epochs 15 \
            --save_last_k 3 \
            --n_runs 1 \
            --clear_model \
            --save_best \
            --data_gene \
            --data_gene_extract \
            --data_gene_extract_epochs 20 \
            --data_gene_epochs 20 \
            --init_tag english \
            --do_train \
            --do_eval \
            --use_same_model \
            --data_gene_wt_constrained \
            --model_filter \
            --model_filter_skip_none \
            --train_by_pair \