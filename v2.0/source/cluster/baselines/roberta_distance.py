import pickle
import argparse
from tqdm import tqdm
from typing import List

import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel

from utils import load_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True
    )
    return parser.parse_args()


@torch.no_grad()
def roberta_inference(text1: str, text2: str) -> float:
    encoded_input1 = tokenizer(text1, return_tensors='pt')
    encoded_input2 = tokenizer(text2, return_tensors='pt')
    output1 = model(**encoded_input1)['pooler_output']
    output2 = model(**encoded_input2)['pooler_output']
    sim_score = metric(output1, output2)
    return sim_score


def pairwise_distances(candidates: List[str]) -> float:
    max_score = 0
    for i in range(len(candidates) - 1):
        pivot_entity = candidates[i].strip()
        other_entity_lst = [ele.strip() for ele in candidates[i+1:]]

        for other_entity in other_entity_lst:
            sim_score = roberta_inference(pivot_entity, other_entity)
            if sim_score > max_score:
                max_score = sim_score
    return max_score


def main():
    global tokenizer, model, metric
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large')
    metric = nn.CosineSimilarity(dim=1)

    args = get_args()
    data = load_json(args.data_path)
    THRESHOLD = .98

    pred = []
    for entry in tqdm(data.values()):
        entity_info = entry['clustered_entities']
        for entity in entity_info:
            candidates = entity.split('|')
            if len(candidates) == 2:
                if (candidates[0] in candidates[1]) or (candidates[1] in candidates[0]):
                    pred.append(1)
                else:
                    min_score = pairwise_distances(candidates)

                    if min_score < THRESHOLD:
                        pred.append(0)
                    else:
                        pred.append(1)

            else:
                min_score = pairwise_distances(candidates)

                if min_score < THRESHOLD:
                    pred.append(0)
                else:
                    pred.append(1)

    print(pred)
    with open('../../data/entity-cluster/results/roberta-distance-result.pkl', 'wb') as f:
        pickle.dump(pred, f)
    f.close()




if __name__ == '__main__':
    main()
