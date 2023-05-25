import ast
import json
import argparse
from typing import List

from utils import load_json, save_txt, load_txt, clean_steps


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True
    )
    parser.add_argument(
        '--template_path', type=str, required=True
    )
    parser.add_argument(
        '--header_path', type=str, required=False
    )
    parser.add_argument(
        '--save_path', type=str, required=True, help='Path to save the template'
    )
    parser.add_argument(
        '--style', type=str, required=True, help='choose from ["text", "code", "chat"]'
    )
    return parser.parse_args()



def main() -> str:
    args = get_args()
    template_data = load_json(args.data_path)
    template = load_txt(args.template_path)

    if args.style in ['code', 'chat']:
        if args.header_path:
            header = open(args.header_path, 'r').read()
            if args.style == 'code':
                prefix = header + '\n'
            elif args.style == 'chat':
                prefix = [ast.literal_eval(header.strip())] 
        else:
            prefix = ''
    else:
        prefix = ''

    for content in template_data.values():
        cur_goal = content['goal']

        cur_steps = clean_steps(content['steps'])

        if args.style == 'text':
            cur_steps = [f'#{i+1} {content}' for (i, content) in enumerate(cur_steps)]
            cur_steps = '\n'.join(cur_steps)

        if args.style == 'chat':
            cur_steps = ' '.join(cur_steps)

        cur_clustered_entities = content['clustered_entities']
        cur_annotation = content['annotation']

        for annotation, clustered_entity in zip(cur_annotation, cur_clustered_entities):

            if ',' in annotation:
                cur_gt = 'No' if args.style == 'chat' else 'False'
            else:
                cur_gt = 'Yes' if args.style == 'chat' else 'True'

            if args.style == 'code':
                clustered_entity_lst = clustered_entity.split('|')
                clustered_entity = "" 
                for entity in clustered_entity_lst:
                    clustered_entity += "'" + entity.strip() + "', "
                clustered_entity = '[' + clustered_entity[:-2] + ']'
            else:
                clustered_entity = clustered_entity.replace(' | ', ', ')

            if args.style == 'chat':
                cur_goal = cur_goal.replace('"', "'")
                cur_steps = cur_steps.replace('"', "'")
                clustered_entity.replace('"', "'")

            cur_entry = template.replace('{goal}', cur_goal).  \
                                 replace('{steps}', str(cur_steps)).  \
                                 replace('{entities}', clustered_entity.replace(' | ', ', '))
            
            if args.style == 'text':
                prefix += cur_entry + cur_gt + '\n\n\n'
            elif args.style == 'code':
                cur_entry = cur_entry.replace('{annotation}', cur_gt)
                prefix += cur_entry + '\n'
            elif args.style == 'chat':
                cur_entry = cur_entry.replace('{annotation}', cur_gt)
                cur_entry = cur_entry.strip().split('\n')
                cur_entry = [ast.literal_eval(ele) for ele in cur_entry]
                prefix += cur_entry

    if args.style in ['text', 'code']:
        save_txt(args.save_path, prefix)
    else:
        with open(args.save_path, 'w') as f:
            json.dump(prefix, f)
        f.close()


if __name__ == '__main__':
    main()


