import json
import argparse
import numpy as np


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
    data = json.load(open(args.data_path, 'r'))
    np.random.seed(42)

    sample_data = np.random.choice(list(data.keys()), 5, replace=False)
    data = {k: v for (k, v) in data.items() if k in sample_data}

    out_json = {}
    for key, val in data.items():
        cur_goal = val['goal']
        cur_steps = val['steps']
        cur_states = val['states']
        cur_entities = [item['entity'] for item in cur_states]
        cur_entities = [item.split(' | ') for item in cur_entities]
        cur_entities = [list for sublst in cur_entities for list in sublst]

        cur_clusters = [item['entity'] for item in cur_states]
        cur_clusters = [item.replace(' | ', ', ') for item in cur_clusters]
        for idx, item in enumerate(cur_clusters):
            if ',' in item:
                cur_clusters[idx] = '(' + item + ')'
        
        out_json[key] = {
            'goal': cur_goal,
            'steps': cur_steps,
            'entities': cur_entities,
            'clusters': cur_clusters,
        }
    
    with open(args.out_path, 'w') as f:
        json.dump(out_json, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()

