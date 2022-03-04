## Training Hyperparameters.

```Model config GPT2Config {
    architectures: [
      GPT2LMHeadModel
      ],
      attn_pdrop: 0.1,
      bos_token_id: null,
      do_sample: false,
      embd_pdrop: 0.1,
      eos_token_ids: null,
      finetuning_task: null,
      id2label: {
      0: "LABEL_0",
      1: "LABEL_1"
      },
      initializer_range: 0.02,
      is_decoder: false,
      label2id: {
      LABEL_0: 0,
      LABEL_1: 1
      },
      layer_norm_epsilon: 1e-05,
      length_penalty: 1.0,
      max_length: 20,
      model_type: "gpt2",
      n_ctx: 1024,
      n_embd: 1024,
      n_head: 16,
      n_layer: 24,
      n_positions: 1024,
      n_special: 0,
      num_beams: 1,
      num_labels: 2,
      num_return_sequences: 1,
      output_attentions: false,
      output_hidden_states: false,
      output_past: true,
      pad_token_id: null,
      predict_special_tokens: true,
      pruned_heads: {},
      repetition_penalty: 1.0,
      resid_pdrop: 0.1,
      summary_activation: null,
      summary_first_dropout: 0.1,
      summary_proj_to_labels: true,
      summary_type: "cls_index",
      summary_use_proj: true,
      temperature: 1.0,
      top_k: 50,
      top_p: 1.0,
      torchscript: false,
      use_bfloat16: false,
      vocab_size: 50257
}
```

## Command used to train.
```
python run_trackworld.py --output_dir=/training_output --model_type=gpt2 --continue_from_dir=/continue_from_dir --model_name_or_path=gpt2-medium --do_train --train_data_file=/training_input/train.jsonl --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --overwrite_output_dir --length 100 --block_size 300 --save_total_limit 3 --save_steps 5000 --fp16 --learning_rate 0.00005 --overridden_model_configs '{"resid_pdrop": 0.1, "attn_dropout": 0.1}' --weight_decay 0.0 --num_train_epochs 30 --window_size 1 --seed 42
```

## Hardware level info.
```
device: cuda, 
n_gpu: 1, 
distributed training: False, 
16-bits training: True
```

## Matching scores published in the paper (inference hyperparams)
```
In the paper we used this setting:
With max len = 100
                prec    recall  f1
ExactMetric     8.10    6.38    3.38
BLEUMetric      22.53   17.86   15.45
ROUGEMetric     38.92   33.51   31.49

If you use max len= 400 then, the scores will improve slightly.
We will revise the arxiv with this information.
                prec    recall  f1
ExactMetric     8.82    7.00    4.13
BLEUMetric      22.85   19.08   16.38
ROUGEMetric     39.54   35.18   32.66
```

There is a small change in the evaluation script which performs deduplication.  This drops the score published in the paper by 0.05 points.
Relevant pull request is here: https://github.com/allenai/openpi-dataset/pull/14



## Commands to run.

Please refer to the README for commands to train and run inference. 
https://github.com/allenai/openpi-dataset/blob/main/README.md


