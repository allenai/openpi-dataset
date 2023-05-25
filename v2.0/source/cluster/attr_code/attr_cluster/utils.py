import json
import numpy as np
from typing import Dict, List, Tuple


def load_json(data_path: str) -> Dict[str, str]:
    with open(data_path, "r") as f:
        return json.load(f)


def clean_steps(steps: List[str]) -> List[str]:
    for i, step in enumerate(steps):
        step = step.strip()
        if not (last_char := step[-1]).isalnum():
            step = step.replace(last_char, '.')
            steps[i] = step
    return steps


def save_json(data_path, data):
    with open(data_path, "w") as f:
        json.dump(data, f, indent=4)


def get_name_from_path(path: str, postfix='.json') -> str:
    if "train" in path.strip().lower():
        return "train" + postfix
    elif "val" in path.strip().lower():
        return "val" + postfix
    else:
        return "test" + postfix


def load_txt(path: str) -> str:
    return open(path, "r").read()


def save_txt(path: str, data: str) -> None:
    with open(path, "w") as f:
        f.write(data)
    f.close()
        

def remove_cluster(openpi_entry: Dict[str, Dict[str, str]]) -> None:
    cur_goal = openpi_entry["goal"]
    cur_steps = openpi_entry["steps"]
    cur_annotations = openpi_entry["states"]
    all_attr = [[] for _ in range(len(cur_steps))]
    all_pre = [[] for _ in range(len(cur_steps))]
    all_post = [[] for _ in range(len(cur_steps))]
    all_entities = []

    for annotation in cur_annotations:
        cur_entities = [ele.strip() for ele in annotation["entity"].split("|")]
        all_entities += cur_entities
        cur_answers = annotation["answers"]

        for i in range(1, len(cur_steps) + 1):
            if cur_info_lst := cur_answers.get(f"step{i}", False):
                for cur_info in cur_info_lst:
                    all_attr[i - 1] += [
                        ele.strip() for ele in cur_info["attribute"].split("|")
                    ]
                    all_pre[i - 1] += [
                        ele.strip() for ele in cur_info["before"].split("|")
                    ]
                    all_post[i - 1] += [
                        ele.strip() for ele in cur_info["after"].split("|")
                    ]

    all_attr = [list(set(sub_lst)) for sub_lst in all_attr]
    all_pre = [list(set(sub_lst)) for sub_lst in all_pre]
    all_post = [list(set(sub_lst)) for sub_lst in all_post]
    all_entities = list(set(all_entities))

    new_annotations = {f'step{i+1}': {} for i in range(len(cur_steps))}
    new_annotations['goal'] = cur_goal

    for i, step in enumerate(cur_steps):
        new_annotations[f'step{i+1}']['step'] = step
        new_annotations[f'step{i+1}']['entity_states'] = []

    for annotation in cur_annotations:
        cur_entities = [ele.strip() for ele in annotation["entity"].split("|")]
        cur_answers = annotation["answers"]

        for i in range(1, len(cur_steps) + 1):
            if cur_info_lst := cur_answers.get(f"step{i}", False):
                for cur_info in cur_info_lst:
                    cur_attr = [ele.strip() for ele in cur_info["attribute"].split("|")]
                    cur_pre = [ele.strip() for ele in cur_info["before"].split("|")]
                    cur_post = [ele.strip() for ele in cur_info["after"].split("|")]

                    other_attr = [ele for ele in all_attr[i-1] if ele not in cur_attr]
                    other_pre = [ele for ele in all_pre[i-1] if ele not in cur_pre]
                    other_post = [ele for ele in all_post[i-1] if ele not in cur_post]
                    other_entities = [ele for ele in all_entities if ele not in cur_entities]

                    cur_dict = {
                        "entity": cur_entities,
                        "attribute": cur_attr,
                        "precondition": cur_pre,
                        "postcondition": cur_post,
                        "other_entity": other_entities,
                        "other_attribute": other_attr,
                        "other_precondition": other_pre,
                        "other_postcondition": other_post
                    }

                new_annotations[f'step{i}']['entity_states'].append(cur_dict)

    return new_annotations


def sample_data(data: Dict[str, Dict[str, str]], num_samples: str=10) -> Dict[str, Dict[str, str]]:
    valid_keys = [k for k in list(data.keys()) if data[k]['clustered_entities']]
    sampled_ids = np.random.choice(len(valid_keys), num_samples, replace=False) 
    sampled_keys = [valid_keys[idx] for idx in sampled_ids]

    sampled_data = {k: data[k] for k in sampled_keys}
    return sampled_data

