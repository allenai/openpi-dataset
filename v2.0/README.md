# OpenPI2.0

The goal of this project is to augment the original OpenPI dataset in the following aspects:
- Canonicalization: add entity & attribute clustering based on GPT
- Salience: add automatic and manual metrics for salience of entities and attributes

## Canonicalization
Taking the dev set as an example, we start with `data/dev-ranked.json` which is the original OpenPI data, with some updates from the one in the original paper (for example, the S-Score is already calculated). See the README in `source/cluster` for performing clustering.

The resulting data file with entity and attribute clusters is `data/dev-data-reformatted-v4.json`.


### Evaluation
To get model predictions based on the canonicalized OpenPI2.0:
- Running `source/predict_schema.py --model MODEL --prompt 1|2` and `predict_states.py --model MODEL` produces predictions for the schemata subtask and the states subtask. The output is for example `data/dev_schema_chatgpt_1.json`. The prompt type 1 corresponds to predicting entities and attributes individually, while the prompt 2 corresponds to the combined prediction of an entire sentence (attribute of entity was pre-state before and post-state after) just like the original OpenPI evaluation.
- Running `source/evaluate_schema.py --model MODEL [--og]` or similarly `evaluate_states.py` and `evaluate_combined.py` performs evaluation of the above settings. `--og` specifies to use the over-generated and expanded clusters for a fairer exact match evaluation.

## Salience
As described in the paper, there are 4 sets of entity salience scores.

### Human annotations
- `data/dev-data-reformatted-v4_votes_salience_1-20.json` contains human annotations by human A
- `data/dev-data-reformatted-v4_votes_salience_1-20_human2.json` contains human annotations by human B

### Votes
- `data/xx_json_dev_without_image.jsonl` is the mturk responses
- Running `source/parse_mturk.py` parses `data/xx_json_dev_without_image.jsonl` and produces `data/proc_entities.json` and `data/proc_attributes.json`
- Running `source/add_votes.py` parses the above two JSON files and adds the mturk votes to `data/dev-data-reformatted-v4.json`, producing `data/dev-data-reformatted-v4_votes.json`

### S-Score
- Running `source/add_scores.py` adds S-Scores to `data/dev-data-reformatted-v4.json`, producing `data/dev-data-reformatted-v4_votes_score_salience_1-20.json`

### LM prediction
- Running `source/predict_salience.py --model MODEL` produces `data/dev-data-reformatted-v4_pred-salience.json` which contains LM-predicted salience scores

### Evaluation
- Running `source/evaluate_salience.py` calculates correlation among the above scores.
- Running `source/plot_correlation.py` plots a bar chart of correlations for the first 20 procedures in the development set.


## Citation
If you find our work helpful, please cite
```
@misc{zhang2023openpi20,
      title={OpenPI2.0: An Improved Dataset for Entity Tracking in Texts}, 
      author={Li Zhang and Hainiu Xu and Abhinav Kommula and Niket Tandon and Chris Callison-Burch},
      year={2023},
      eprint={2305.14603},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
