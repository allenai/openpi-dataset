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

    entity_attr_template = lambda ent, attr: f'{attr} of {ent}'

    out_json = {}
    for key, val in data.items():
        cur_goal = val['goal']
        cur_steps = val['steps']
        cur_states = val['states']

        cur_attr_lst = []
        cur_cluster_attr_lst = []
        for step_entry in cur_states:
            cur_entity = step_entry['entity'].split('|')[0].strip()
            for step_value in step_entry['answers'].values():
                cur_attr = [item['attribute'].split('|') for item in step_value]
                cur_attr = [lst for sublst in cur_attr for lst in sublst]
                cur_attr = [item.strip() for item in cur_attr]
                cur_attr = [entity_attr_template(cur_entity, item) for item in cur_attr]
                cur_attr_lst += cur_attr

                cur_cluster = [item['attribute'].replace(' | ', ', ') for item in step_value]
                for j, item in enumerate(cur_cluster):
                    if ',' not in item:
                        cur_cluster[j] = entity_attr_template(cur_entity, item)
                    if ',' in item:
                        cur_cluster[j] = ', '.join([entity_attr_template(cur_entity, attr) for attr in item.split(',')])
                
                cur_cluster_attr_lst += cur_cluster
                cur_cluster_attr_lst = [item.strip() for item in cur_cluster_attr_lst]
        
        out_json[key] = {
            'goal': cur_goal,
            'steps': cur_steps,
            'attributes': list(set(cur_attr_lst)),
            'clusters': list(set(cur_cluster_attr_lst)),
        }
    
    with open(args.out_path, 'w') as f:
        json.dump(out_json, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()