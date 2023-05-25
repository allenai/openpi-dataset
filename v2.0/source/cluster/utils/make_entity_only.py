''' 
script to convert procedure to entity only
'''

import json 
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the data'
    )
    return parser.parse_args()


def plural_to_singular(word: str) -> str:
    if len(word) > 4 and word[-2:] != 'ss':
        if word.endswith("ies"):
            return word[:-3] + "y"
        elif word.endswith("es"):
            return word[:-2]
        elif word.endswith("s"):
            return word[:-1]
        else:
            return word
    else:
        return word


def main():
    args = get_args()
    data = json.load(open(args.data_path, 'r'))
    
    out_data = {}
    for key, val in data.items():
        cur_goal, cur_steps = val['goal'], val['steps']
        cur_states = val['states']
        cur_entities = [item['entity'].split(' | ') for item in cur_states]
        cur_entities_flattened = [item for sublist in cur_entities for item in sublist]

        out_data[key] = {
            'goal': cur_goal,
            'steps': cur_steps,
            'entities': cur_entities,
            'entities_flattened': cur_entities_flattened
        }
    
    fname = args.data_path.split('/')[-1].split('.')[0]
    json.dump(out_data, open(f'../data/{fname}_entity.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
