#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python -m llava.eval.model_vqa_loader \
    --model-path /data/LLaVA/llava-v1.5-7b \
    --question-file /home/gxxl/ywh/LLaVA_edit/playground/data/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data/LLaVA/data/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --reduction_ratio 60 \


python -m llava.eval.eval_textvqa \
    --annotation-file /home/gxxl/ywh/LLaVA_edit/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl

