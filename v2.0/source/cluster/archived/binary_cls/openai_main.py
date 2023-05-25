import os
import ast
import json
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from openai_inference import gpt3_inference, chat_inference
from utils import load_json, sample_data, save_json, load_txt, clean_steps

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the un-clustered file'
    )
    parser.add_argument(
        '--template_path', type=str, required=True, help='path to the prompt'
    )
    parser.add_argument(
        '--examples_path', type=str, required=True, help='path to in-context examples'
    )
    parser.add_argument(
        '--model_name', type=str, default='gpt-3.5-turbo', help='model name for GPT3'
    )
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the api key'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='entity', 
        help='specify which part of the schema to fix. choose from ["entity", "attribute", "precond", "postcond"]'
    )
    return parser.parse_args()


def load_template(template_path: str) -> str:
    return ''.join(open(template_path, 'r').readlines())


def save_result(results: List[str], path: str):
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    f.close()


def main():
    args = get_args()
    data = load_json(args.data_path)
    examples = load_txt(args.examples_path)
    template = load_txt(args.template_path)

    if 'chat' in args.examples_path:
        examples = load_json(args.examples_path)
    else:
        examples = load_txt(args.examples_path)

    openai.api_key_path = args.api_path

    results = []
    for entry in tqdm(data.values()):
        cur_goal, cur_steps, cur_entities, cur_annotations = entry.values()

        cur_steps = ' '.join(cur_steps)

        for entity_info, gt_info in zip(cur_entities, cur_annotations):

            entity_info_lst = entity_info.split('|')
            clustered_entity = "" 
            for entity in entity_info_lst:
                clustered_entity += "'" + entity.strip() + "', "
            entity_info = '[' + clustered_entity[:-2] + ']'
            
            cur_template = template.replace('{goal}', cur_goal)  \
                                   .replace('{steps}', cur_steps)  \
                                   .replace('{entities}', entity_info)

            cur_template = '[' + cur_template.strip().split('\n')[0] + ']'
            cur_template = ast.literal_eval(cur_template)
            # cur_input = [examples[0]] + cur_template
            cur_input = examples + cur_template
            print(cur_input)
            raise SystemExit()
            out = chat_inference(cur_input, args.model_name)
            out = out['content']
            
            results.append(out)

    save_result(results, os.path.join('../../../data/entity-cluster/results/', 'turbo-binary-result.pkl'))

if __name__ == '__main__':
    main()
