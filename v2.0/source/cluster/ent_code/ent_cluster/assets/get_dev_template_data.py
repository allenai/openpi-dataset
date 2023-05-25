import os
import argparse
import numpy as np

from utils import sample_data, load_json, save_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True
    )
    parser.add_argument(
        '--out_path', type=str, required=True
    )
    return parser.parse_args()


def main():
    np.random.seed(42)
    args = get_args()
    data = load_json(args.data_path)

    template_data = sample_data(data, 5)
    dev_data = sample_data(data, 20)

    for key in template_data.keys():
        template_data[key]['annotation'] = template_data[key]['clustered_entities']

    for key in dev_data.keys():
        dev_data[key]['annotation'] = dev_data[key]['clustered_entities']

    template_path = os.path.join(args.out_path, 'template-data.json')
    dev_path = os.path.join(args.out_path, 'dev-data.json')

    save_json(template_path, template_data)
    save_json(dev_path, dev_data)

if __name__ == '__main__':
    main()


