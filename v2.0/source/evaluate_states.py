import json
import numpy as np
import itertools
import argparse
import bert_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='Either davinci or chatgpt.')

args = parser.parse_args()

acc_scores = []
bert_scores = []
preds = []
golds = []
with open(f"../data/dev_states_{args.model}.json") as f_pred, open(f"../data/dev_states_gold.json") as f_gold:
    pred_data = json.load(f_pred)
    gold_data = json.load(f_gold)
    for (id, pred_steps), (id, gold_steps) in zip(pred_data.items(), gold_data.items()):
        for pred_step, gold_step in zip(pred_steps, gold_steps):
            for pred_state, gold_state in zip(pred_step, gold_step):
                pred_before = pred_state[2]
                pred_after = pred_state[3]
                gold_befores = gold_state[2].split(" | ")
                gold_afters = gold_state[3].split(" | ")
                # acc
                if pred_before in gold_befores and pred_after in gold_afters:
                    acc_scores.append(1)
                else:
                    acc_scores.append(0)
                # bs
                preds.append(pred_before)
                golds.append(gold_befores[0])
                preds.append(pred_after)
                golds.append(gold_afters[0])

print(np.mean(acc_scores))
P, R, F1 = bert_score.score(cands = preds, refs = golds, model_type = "microsoft/deberta-xlarge-mnli")
print(np.mean(F1.numpy()))