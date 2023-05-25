import os
import argparse

from utils import load_json, save_json, remove_cluster, get_name_from_path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to data'
    )
    parser.add_argument(
        '--output_path', type=str, required=True, help='path to save output'
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = load_json(args.data_path)
    
    out_data = {}

    for key, entry in data.items():
        new_entry = remove_cluster(entry)
        out_data[key] = new_entry
    
    fname = get_name_from_path(args.data_path)
    out_path = os.path.join(args.output_path, fname)
    save_json(out_path, out_data)

if __name__ == '__main__':
    main()

