GOLD_CSV=${1-}
PRED_CSV=${2-}
PREFIX_NAME_CSV=${3-}
ONE_METRICSJSON=${4-}

set -x # print the command being executed.
set -e

IFS=',' read -ra golds <<<"$GOLD_CSV"
IFS=',' read -ra preds <<<"$PRED_CSV"
IFS=',' read -ra eval_names <<<"$PREFIX_NAME_CSV"

printf "{" > "$ONE_METRICSJSON"

first_done=false

for i in ${!golds[*]}; do

  if [ "$first_done" = true ] ; then
    printf ", " >> "$ONE_METRICSJSON"  # an extra trailing comma in json is still valid and is ignored.
  fi

  if [ "$first_done" = false ] ; then
    first_done=true
  fi

  curr_output=$(
    python src/eval/tw_specific_eval/simple_eval.py \
      --quiet \
      -g "${golds[i]}" \
      -p "${preds[i]}" \
      -n "${eval_names[i]}"
  )
  printf "%s" "$curr_output" >> "$ONE_METRICSJSON"


done

# The following will close the file.
echo "}" >> "$ONE_METRICSJSON"
