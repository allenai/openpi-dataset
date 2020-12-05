# can increase these args as needed.
# current args order: [model_configs_*keyvalue_csv] [learning_rate] [weight_decay]
MODEL_CONFIGS=${1-}  # e.g., '{"resid_pdrop": 0.1, "attn_dropout": 0.1}'
LR=${2-0.00002} # previously 2e-5
WEIGHT_DECAY=${3-0.0}
MAX_LEN=${4-20}
NUM_EPOCHS=${5-20}
MODEL_TYPE=${6-gpt2}
BATCH_SIZE_TRAIN=${7-8}
BATCH_SIZE_EVAL=${8-16}
BLOCK_SIZE=${9-512}

set -x  # print the command being executed.

#    --model_name_or_path=gpt2-medium \
#    --block_size 512 \
#    --continue_from_dir=/continue_from_dir \
#    --length 12 or 50 \
#    --block_size 300 or 512 \
#    --do_twbench \
#    --test_twbench_data_file=data/trackworld/tw_bench/tw_bench_propara_npn_ea.jsonl \
#    --per_gpu_train_batch_size 8 or 4 \
#    --per_gpu_eval_batch_size 16 \
#    --do_twbench \
#    --test_twbench_data_file=data/trackworld/tw_bench/tw_bench_propara_npn_ea.jsonl \


# The following is for ea (enable tw_bench if needed).
#    --do_twbench \
#    --test_twbench_data_file=data/trackworld/tw_bench/tw_bench_propara_npn_ea.jsonl \

#    --do_eval \
#    --eval_data_file=data/trackworld/simplified/ea_positive/dev.jsonl \
#    --per_gpu_eval_batch_size 16 \

# This is for upperbound.
#    --continue_from_dir=/continue_from_dir \
python training/run_trainer.py \
    --output_dir=/tmp/training_output \
    --model_type="$MODEL_TYPE" \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=data/formatted_for_gpt2/dev.jsonl \
    --per_gpu_train_batch_size $BATCH_SIZE_TRAIN \
    --per_gpu_eval_batch_size $BATCH_SIZE_EVAL \
    --overwrite_output_dir \
    --length $MAX_LEN \
    --block_size $BLOCK_SIZE \
    --save_total_limit 3 \
    --save_steps 5000 \
    --learning_rate $LR \
    --overridden_model_configs "$MODEL_CONFIGS" \
    --weight_decay $WEIGHT_DECAY \
    --num_train_epochs $NUM_EPOCHS
#    --seed 42
#    --fp16 \


#python run_trackworld.py \
#    --output_dir=/training_output \
#    --model_type=gpt2 \
#    --continue_from_dir=/continue_from_dir \
#    --model_name_or_path=gpt2-medium \
#    --do_train \
#    --train_data_file=/training_input/train.jsonl \
#    --per_gpu_train_batch_size 1 \
#    --per_gpu_eval_batch_size 1 \
#    --overwrite_output_dir \
#    --length 60 \
#    --block_size 1024 \
#    --save_total_limit 3 \
#    --save_steps 5000 \
#    --learning_rate $LR \
#    --overridden_model_configs "$MODEL_CONFIGS" \
#    --weight_decay $WEIGHT_DECAY \
#    --num_train_epochs 1 \
#    --fp16 \
#    --do_sent_window \
#    --seed 42


# The following is for e2e
#    --do_eval \
#    --eval_data_file=data/trackworld/simplified/e2e_random_positive/dev.jsonl \
#    --do_test \
#    --test_data_file=data/trackworld/simplified/e2e_random_positive/test.jsonl \
#    --continue_from_dir=/continue_from_dir \
#python run_trackworld.py \
#    --output_dir=/training_output \
#    --model_type=gpt2 \
#    --model_name_or_path=gpt2-medium \
#    --do_train \
#    --train_data_file=data/trackworld/simplified/e2e_ordered_positive/train.jsonl \
#    --per_gpu_train_batch_size 4 \
#    --per_gpu_eval_batch_size 8 \
#    --overwrite_output_dir \
#    --length 50 \
#    --block_size 500 \
#    --save_total_limit 3 \
#    --save_steps 5000 \
#    --fp16 \
#    --learning_rate $LR \
#    --overridden_model_configs "$MODEL_CONFIGS" \
#    --weight_decay $WEIGHT_DECAY \
#    --num_train_epochs 5 \
#    --seed 42


