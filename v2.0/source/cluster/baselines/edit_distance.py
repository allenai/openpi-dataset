import pickle
import argparse
from typing import List
from utils import load_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True
    )
    return parser.parse_args()


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def pairwise_distances(candidates: List[str]) -> float:
    max_score = 0
    for i in range(len(candidates) - 1):
        pivot_entity = candidates[i].strip()
        other_entity_lst = [ele.strip() for ele in candidates[i+1:]]

        for other_entity in other_entity_lst:
            sim_score = levenshteinDistance(pivot_entity, other_entity)
            if sim_score > max_score:
                max_score = sim_score
    return max_score


def main():
    args = get_args()
    data = load_json(args.data_path)
    THRESHOLD = 10

    pred = []
    for entry in data.values():
        entity_info = entry['clustered_entities']
        for entity in entity_info:
            candidates = entity.split('|')
            if len(candidates) == 2:
                if (candidates[0] in candidates[1]) or (candidates[1] in candidates[0]):
                    pred.append(1)
                else:
                    max_score = pairwise_distances(candidates)

                    if max_score > THRESHOLD:
                        pred.append(0)
                    else:
                        pred.append(1)

            else:
                max_score = pairwise_distances(candidates)

                if max_score > THRESHOLD:
                    pred.append(0)
                else:
                    pred.append(1)

    with open('../../data/entity-cluster/results/edit-distance-result.pkl', 'wb') as f:
        pickle.dump(pred, f)
    f.close()


if __name__ == "__main__":
    main()
