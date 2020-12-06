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