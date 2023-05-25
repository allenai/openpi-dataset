import os
import ast
import json
import argparse
from typing import List

from utils import load_json, save_txt, load_txt, clean_steps, save_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True
    )
    parser.add_argument(
        '--template_path', type=str, required=True
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='Path to save the template'
    )
    return parser.parse_args()


def main() -> str:
    args = get_args()
    template_data = load_json(args.data_path)
    template = load_txt(args.template_path)


    out_examples = ''

    for entry in template_data.values():
        cur_goal = entry['goal']
        cur_steps = ' '.join(entry['steps'])
        cur_entity_states_lst = entry['entity_states']

        for cur_entity_state in cur_entity_states_lst:
            cur_openpi_narratives = cur_entity_state['openpi_narrative']

            for narrative in cur_openpi_narratives:
                cur_example = template.replace('{goal}', cur_goal.replace('"', "'"))  \
                                      .replace('{steps}', cur_steps.replace('"', "'"))  \
                                      .replace('{openpi_template}', narrative.strip().replace('"', "'"))

                out_examples += cur_example + '\n\n'
                print(out_examples)
                raise SystemExit()

    # with open(args.out_path, 'w') as f:
    #     f.write(out_examples)
    # f.close()


if __name__ == '__main__':
    main()


