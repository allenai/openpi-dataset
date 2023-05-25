import os
import ast
import json
import argparse
from typing import List

from utils import load_json, load_txt, save_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path', type=str, required=True, help='Path to save the template'
    )
    return parser.parse_args()


def main() -> str:
    args = get_args()
    data = load_json('../../data/example_data_annotated.json')
    template = load_txt('./assets/templates/chat_template.txt')
    openpi_template = lambda attr, ent, pre, post: f'The {attr} of {ent} is {pre} before and {post} afterwards.'

    out_dict = {}
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
            tt = 0
            if len(cur_entities) > 1:
                cur_alternative_narrative = [openpi_template(cur_attr, item.strip(), cur_pre, cur_post) for item in cur_entities[1:]]
                if len(cur_alternative_narrative) > 1:
                    tt = 1
                cur_alternative_narrative = '- ' + '\\n- '.join(cur_alternative_narrative)
            else:
                cur_alternative_narrative = 'None'

            cur_template = template.replace('{procedure_context}', proc_context)  \
                                   .replace('{entity_context}', cur_narrative)  \
                                   .replace('{answer}', cur_alternative_narrative)
            
            cur_template = cur_template.strip().split('\n')
            cur_template = [ast.literal_eval(item) for item in cur_template]
            out_dict[f'{proc_id}_{state_id}'] = cur_template

    save_json(args.out_path, out_dict)


if __name__ == '__main__':
    main()


