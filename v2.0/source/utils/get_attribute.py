import json 
import argparse


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
    args = get_args()
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    f.close()

    out_data = []
    for content in data.values():
        entity_state = [{'entity': item['entity'], 'openpi_narrative': item['openpi_narrative']} for item in content['entity_states']]
        cur_dict = {
            'goal': content['goal'],
            'steps': content['steps'],
            'entity_state': entity_state
        }
        out_data.append(cur_dict)
    
    with open(args.out_path, 'w') as f:
        json.dump(out_data, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()
        