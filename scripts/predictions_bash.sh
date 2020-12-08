TRAINED_MODEL=${1-}
INPUT_TO_PRED_CSV=${2-}
OUTPUT_FILEPATH_CSV=${3-}
AGG_OUTPUT_FILEPATH_CSV=${4-}
MAX_LEN=${5-20}

set -x  # print the command being executed.

IFS=',' read -ra prediction_input_files <<< "$INPUT_TO_PRED_CSV"
IFS=',' read -ra prediction_output_files <<< "$OUTPUT_FILEPATH_CSV"
IFS=',' read -ra agg_pred_output_files <<< "$AGG_OUTPUT_FILEPATH_CSV"

for i in ${!prediction_input_files[*]}; do
  python training/generation.py \
      --model_path "$TRAINED_MODEL" \
      --test_input_file "${prediction_input_files[i]}" \
      --test_output_file "${prediction_output_files[i]}" \
      --test_output_agg_file "${agg_pred_output_files[i]}" \
      --max_len $MAX_LEN
done