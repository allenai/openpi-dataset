import argparse

from utils import load_json, save_json, get_name_from_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to OpenPI data'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to save the parsed data'
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = load_json(args.data_path)

    out_dict = {}
    counter_cluster, counter_all = 0, 0
    for key, val in data.items():
        new_val = {
            'goal': val['goal'],
            'steps': val['steps']
        }

        cur_states = val['states']
        clustered_entities = [d['entity'] for d in cur_states if '|' in d['entity']]

        counter_cluster += len(clustered_entities)
        counter_all += len(cur_states)

        new_val['clustered_entities'] = clustered_entities
        out_dict[key] = new_val

    print(f'{counter_cluster}/{counter_all} states have clustered entity')
    save_json(args.out_path, out_dict)


if __name__ == '__main__':
    main()

