import os
import re
import ast
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from transformers import GPT2Tokenizer

from openai_inference import gpt3_inference, chat_inference
from utils import load_json, save_json, load_template, load_txt


def get_input_token_count(input: list, tokenizer: callable) -> int:
    count = 0
    for d in input:
        count += len(tokenizer(d['content']).input_ids)
    return count


def make_example(example_candidates: dict, tokenizer: callable):
    MAX_TOKEN = 1500 

    out_example = []
    total_token = 0
    visited_idx = []
    while total_token < MAX_TOKEN:
        sample_idx = np.random.choice(list(example_candidates.keys()), 1)[0]
        if sample_idx not in visited_idx:
            visited_idx.append(sample_idx)
        else:
            continue
        cur_example = example_candidates[sample_idx]
        out_example += cur_example
        total_token += get_input_token_count(cur_example, tokenizer)
    
    return out_example


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the file that stores OpenAI API key'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to save the gpt3 output'
    )
    return parser.parse_args()


def main():
    np.random.seed(42)
    global args
    args = get_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    openai.api_key_path = args.api_path

    data = load_json('../../data/dev_data_mini.json')
    examples = load_json('./assets/examples/chat_examples.json')
    template = load_txt('./assets/templates/chat_template.txt')
    openpi_template = lambda attr, ent, pre, post: f'The {attr} of {ent} is {pre} before and {post} afterwards.'
    header = [{"role": "system", "content": "You are a distinguished professor in English Literature."}]

    examples = header + make_example(examples, tokenizer)

    # regular expression to extract keywords
    extract_template = re.compile('The (.*) of (.*) is (.*) before and (.*) afterwards.')

    out_data = {}
    for proc_id, content in data.items():
        cur_goal, cur_steps = content['goal'], content['steps']
        cur_states = content['states']

        proc_context = f'I am trying to {cur_goal.lower()}.'
        for idx, step in enumerate(cur_steps):
            if idx == 0:
                proc_context += f' First, I {step}'
            else:
                proc_context += f' Then, I {step}'
        proc_context += ' Do you get it?'
        
        for state_id, state in enumerate(cur_states):
            cur_entities = state['entity'].split('|')
            cur_answers = state['answers']
            for ans in cur_answers.values():
                if ans:
                    cur_attr = ans[0]['attribute'].split('|')[0].strip()
                    cur_pre = ans[0]['before'].split('|')[0].strip()
                    cur_post = ans[0]['after'].split('|')[0].strip()
                    break
            
            cur_narrative = openpi_template(cur_attr, cur_entities[0].strip(), cur_pre, cur_post)

            cur_template = template.replace('{procedure_context}', proc_context)  \
                                   .replace('{entity_context}', cur_narrative)
            
            cur_template = cur_template.strip().split('\n')
            cur_template = [ast.literal_eval(item) for item in cur_template][:-1]

            cur_input = examples + cur_template

            out = chat_inference(cur_input)
            print(out)
    







if __name__ == '__main__':
    main()

