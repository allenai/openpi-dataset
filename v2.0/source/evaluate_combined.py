import bert_score
import json
import numpy as np
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='Either davinci or chatgpt.')
parser.add_argument('--metric', required=True, type=str, help='Either bs or em.')
parser.add_argument('--og', action='store_true', help='Whether to use overgenerated clusters.')

args = parser.parse_args()

if args.og:
    fname = "../data/dev_schema_gold_2_og.json"
else:
    fname = "../data/dev_schema_gold_2.json"

def bs():
    with open(f"../data/dev_schema_{args.model}_2.json") as f_pred, open(fname) as f_gold:
        all_scores = []
        pred_data = json.load(f_pred)
        gold_data = json.load(f_gold)
        for (id, pred_steps), (id, gold_steps) in zip(pred_data.items(), gold_data.items()):
            for pred_step, gold_step in zip(pred_steps, gold_steps):
                for pred_sent in pred_step:
                    pred_sent_list = []
                    gold_sent_list = []
                    for gold_sents in gold_step:
                        #P, R, F1 = bert_score.score(cands = [pred_sent for i in range(len(gold_sents))], refs = gold_sents, model_type = "microsoft/deberta-xlarge-mnli")
                        pred_sent_list += [pred_sent for i in range(len(gold_sents))]
                        gold_sent_list += gold_sents
                    if pred_sent_list and gold_sent_list:
                        P, R, F1 = bert_score.score(cands = pred_sent_list, refs = gold_sent_list, model_type = "microsoft/deberta-xlarge-mnli")
                        max_F1 = np.max(F1.numpy())
                        all_scores.append(max_F1)
            #break
            print(id)
            print(np.mean(all_scores))

    print(np.mean(all_scores))

def em():
    with open(f"../data/dev_schema_{args.model}_2.json") as f_pred, open("../data/dev_schema_gold_2.json") as f_gold:
        all_scores = []
        pred_correct = 0
        pred_total = 0
        gold_correct = 0
        gold_total = 0



        pred_data = json.load(f_pred)
        gold_data = json.load(f_gold)
        for (id, pred_steps), (id, gold_steps) in zip(pred_data.items(), gold_data.items()):
            for pred_step, gold_step in zip(pred_steps, gold_steps):
                gold_total += len(gold_step)
                found_gold_indices = set()
                for pred_sent in pred_step:
                    pred_total += 1
                    for i, gold_step_sents in enumerate(gold_step):
                        for gold_sents in gold_step_sents:
                            if pred_sent in gold_sents:
                                pred_correct += 1
                                found_gold_indices.add(i)
                                break
                        else:
                            continue
                        break
                gold_correct += len(found_gold_indices)

                    
            #break
    
    print(pred_correct, pred_total, gold_correct, gold_total)

    precision = pred_correct / pred_total
    recall = gold_correct / gold_total
    F1 = 2 * precision * recall / (precision + recall)
    
    print(F1)

if args.metric == "bs":
    bs()
elif args.metric == "em":
    em()