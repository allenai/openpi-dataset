# openpi-dataset
OpenPI dataset for tracking entities in open domain procedural text
(EMNLP 2020)

`Paper`: https://www.semanticscholar.org/paper/A-Dataset-for-Tracking-Entities-in-Open-Domain-Text-Tandon-Sakaguchi/d47ad0a606bedf41dcea614bfa7b7494879c7ba0#extracted 

arxiv: https://arxiv.org/pdf/2011.08092.pdf

`Project page`: https://allenai.org/data/openpi

`Evaluation`: ```python eval/simple_eval.py -g data/gold/test/id_answers.jsonl -p /path/to/predictions_file --quiet``` (no diagnostics file generated when using --quiet)
e.g., ```python eval/simple_eval.py -g data/gold/test/id_answers.jsonl -p emnlp2020-predictions/predictions/on_openpi_v1.0/test.predictions.jsonl --quiet```

`Dataset` JSON format. There are four files:
  (a) id_question.jsonl: each line is a json with an id, and the input sentence and its past sentences i.e., "x"
  (b) id_question_metadata.jsonl: the metadata corresponding to the question such as topic. Each line is a json with an id, and the metadata
  (c) id_answers_metadata.jsonl: each line is a json with an id, and the a list of answers i.e., "y"
  (d) id_answers.jsonl: the metadata corresponding to the answer. Each line is a json with an id, and the metadata (such as entity, attribute, before value, after value).


`Leaderboard`: coming soon...
