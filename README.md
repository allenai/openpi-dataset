# Openpi-dataset
OpenPI dataset for tracking entities in open domain procedural text
(EMNLP 2020)

![Openpi Task](data/figs/figure-introduction.png)

`Paper`: 
https://www.aclweb.org/anthology/2020.emnlp-main.520.pdf

`Project page`: https://allenai.org/data/openpi

## Dataset

Dataset files are available in JSON format. There are four files:
  - `id_question.jsonl`: each line is a json with an id, and the input sentence and its past sentences i.e., "x"
  - `id_question_metadata.jsonl`: the metadata corresponding to the question such as topic. Each line is a json with an id, and the metadata
  - `id_answers_metadata.jsonl`: each line is a json with an id, and the a list of answers i.e., "y"
  - `id_answers.jsonl`: the metadata corresponding to the answer. Each line is a json with an id, and the metadata (such as entity, attribute, before value, after value).


## Training 


```
sh scripts/trackworld_training_bash.sh
```

## Run Prediction

```
sh scripts/trackworld_predictions_bash.sh
```

## Run Evaluation

```
python eval/simple_eval.py 
    -g data/gold/test/id_answers.jsonl
    -p /path/to/predictions_file 
    --quiet
```
(no diagnostics file generated when using --quiet)


## Leaderboard

coming soon...

## Citation

If you use this dataset in your work, please cite
```
@inproceedings{tandon-etal-2020-dataset,
    title = "A Dataset for Tracking Entities in Open Domain Procedural Text",
    author = "Tandon, Niket  and
      Sakaguchi, Keisuke  and
      Dalvi, Bhavana  and
      Rajagopal, Dheeraj  and
      Clark, Peter  and
      Guerquin, Michal  and
      Richardson, Kyle  and
      Hovy, Eduard",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.520",
    doi = "10.18653/v1/2020.emnlp-main.520",
    pages = "6408--6417"
}
```