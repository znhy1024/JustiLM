#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

size=xl
seed=2
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE=data/train.${seed}.jsonl
EVAL_FILES=data/test.jsonl 
PRETRAINED_MODEL=checkpoints/models/atlas/${size}
PASSAGES=data/copora/docs.jsonl
INDEX=data/copora/index/
SAVE_DIR=exps
FC_FILE=data/copora/fc_articles.json
LABEL_FILE=data/cls_label.json
EXPERIMENT_NAME=atlas-${size}-seed${seed}-lgret-lglm
NCTX=20

torchrun --standalone --nnodes 1 --nproc_per_node 4 train.py \
    --shuffle \
    --fc_only \
    --fc_file ${FC_FILE} \
    --use_gradient_checkpoint_reader \
    --name ${EXPERIMENT_NAME} \
    --precision fp32 \
    --shard_optim --shard_grads \
    --reader_model_type google/t5-${size}-lm-adapt \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --label_file ${LABEL_FILE} \
    --per_gpu_batch_size 1 \
    --n_context ${NCTX} --retriever_n_context ${NCTX} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port ${port} \
    --passages ${PASSAGES}\
    --save_index_path ${INDEX} \
    --seed ${seed}


torchrun --standalone --nnodes 1 --nproc_per_node 4 train.py \
    --shuffle \
    --train_retriever --query_side_retriever_training \
    --use_gradient_checkpoint_retriever \
    --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --lglm \
    --lgret \
    --name ${EXPERIMENT_NAME} \
    --precision fp32 \
    --shard_optim --shard_grads \
    --reader_model_type google/t5-${size}-lm-adapt \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoint/step-100 \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --label_file ${LABEL_FILE} \
    --fc_file ${FC_FILE} \
    --per_gpu_batch_size 1 \
    --n_context ${NCTX} --retriever_n_context ${NCTX} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 100 \
    --main_port ${port} \
    --write_results \
    --passages ${PASSAGES}\
    --load_index_path ${INDEX} \
    --seed ${seed}

python -u src/compute_metric.py -test_file ${EVAL_FILES} -output_file ${SAVE_DIR}/${EXPERIMENT_NAME}/test-result.jsonl -label_file  ${LABEL_FILE}



