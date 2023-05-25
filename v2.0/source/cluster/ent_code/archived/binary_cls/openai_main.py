import os
import ast
import json
import random
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from transformers import GPT2Tokenizer

from openai_inference import chat_inference
from utils import load_json, save_json, load_txt, plural_to_singular


def make_example_lst(examples: dict) -> list:
    yes_counter, no_counter = 0, 0
    yes_ind, no_ind = yes_counter < args.num_shot // 2, no_counter < args.num_shot // 2
    example_lst = []
    while yes_ind or no_ind:
        sampled_example_idx = np.random.choice(list(examples.keys()), 1, replace=False)[0]
        cur_example = examples[sampled_example_idx]
        cur_example = cur_example.strip().split('\n')
        cur_example = [ast.literal_eval(item) for item in cur_example]

        if no_ind and cur_example[-1]['content'] == 'No':
            example_lst += [cur_example]
            no_counter += 1
            no_ind = no_counter < args.num_shot // 2

        elif yes_ind and cur_example[-1]['content'] == 'Yes':
            example_lst += [cur_example]
            yes_counter += 1
            yes_ind = yes_counter < args.num_shot // 2
    
    random.shuffle(example_lst)
    example_lst = [lst for sublst in example_lst for lst in sublst]
    example_lst = [{"role": "system", "content": "You are a Distinguished Professor in English Literature."}] + example_lst
    
    return example_lst


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_shot', type=int, default=10, help='number of shots'
    )
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the api key'
    )
    return parser.parse_args()


def load_template(template_path: str) -> str:
    return ''.join(open(template_path, 'r').readlines())


def save_result(results: List[str], path: str):
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    f.close()


def count_token(inp: str) -> int:
    return len(tokenizer(inp).input_ids)


def count_chat_token(inp: list) -> int:
    return sum([count_token(item['content']) for item in inp])


def compute_cost(token_count: int) -> float:
    return token_count / 1000 * 0.002


def main():
    global args, tokenizer
    args = get_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    openai.api_key_path = args.api_path
    random.seed(96)
    np.random.seed(96)

    data = load_json('../../data/dev_data_mini_entity.json')
    template = load_txt('./assets/templates/entity_chat_template.txt')
    examples = load_json('./assets/examples/entity_chat_example.json')
    examples = make_example_lst(examples)
    
    result_dict = {}
    # total_token = 0
    for proc_id, content in tqdm(data.items(), position=0, leave=False):
        cur_goal = content['goal']
        cur_steps = content['steps']

        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, {step.lower()}')

        cur_steps = cur_step_narratives[-1]

        cur_entities = [plural_to_singular(item) for item in content['entities_flattened']]

        clustered_entities = []
        cur_clusters = {item: [] for item in cur_entities}
        for i in tqdm(range(len(cur_entities)), position=1, leave=False):
            for j in range(len(cur_entities)):
                if i != j:
                    ent1, ent2 = cur_entities[i], cur_entities[j]

                    # skip if already clustered
                    if ent1 in clustered_entities or ent2 in clustered_entities:
                        continue

                    cur_goal = cur_goal.replace('"', "'")
                    cur_steps = cur_steps.replace('"', "'")
                    ent1, ent2 = ent1.replace('"', "'"), ent2.replace('"', "'")

                    cur_template = template.replace('{goal}', cur_goal)  \
                                           .replace('{steps}', cur_steps)  \
                                           .replace('{entities}', f'({ent1}, {ent2})')
                    
                    cur_template = ast.literal_eval(cur_template.strip().split('\n')[0])
                    cur_input = examples + [cur_template]
                    # total_token += count_chat_token(cur_input)

                    out = chat_inference(cur_input)
                    out = out['content']

                    if out.strip().lower() == 'yes':
                        cur_clusters[ent1].append(ent2)
                        clustered_entities.append(ent2)

        for ent in clustered_entities:
            del cur_clusters[ent]

        result_dict[proc_id] = cur_clusters

    with open('./result/entity_chat_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    f.close()
    # print(compute_cost(total_token))


if __name__ == '__main__':
    main()
