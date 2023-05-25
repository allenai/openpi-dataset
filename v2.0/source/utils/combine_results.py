import os
import re
import json
import argparse
from glob import glob


def get_args():
    '''Get arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder', type=str, required=True, help='path to the clustering dataset'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to the output file'
    )
    return parser.parse_args()


def load_json(data_path: str) -> dict:
    '''Load json file.'''
    with open(data_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def main():
    args = get_args()
    data_folder = args.data_folder

    out_data = {}
    data_path_lst = glob(os.path.join(data_folder, '*.json'))
    data_path_lst = [item for item in data_path_lst if 'all-part' in item]

    visited_keys = []
    for data_path in data_path_lst:
        data = load_json(data_path)
        for key, val in data.items():
            if 'gpt3_output' in val.keys():
                if key in visited_keys:
                    raise ValueError(f'Key {key} is visited twice.')
                else:
                    out_data[key] = val
                    visited_keys.append(key)
    
    with open(args.out_path, 'w') as f:
        json.dump(out_data, f, indent=4)
    f.close()

if __name__ == '__main__':
    main()