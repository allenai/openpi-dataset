'''
script to get statistics of clustered entities, attributes, preconditions, and postconditions
'''

import argparse

from utils import load_json, get_name_from_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to OpenPI data'
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = load_json(args.data_path)

    counter_entity_cluster, counter_entity_all = 0, 0
    counter_attr_cluster, counter_attr_all = 0, 0
    counter_pre_cluster, counter_pre_all = 0, 0
    counter_post_cluster, counter_post_all = 0, 0

    for val in data.values():
        cur_states = val['states']
        clustered_entities = [d['entity'] for d in cur_states if '|' in d['entity']]

        counter_entity_cluster += len(clustered_entities)
        counter_entity_all += len(cur_states)

        for state in cur_states:
            for step in state['answers'].values():
                if step:
                    for entry in step:
                        cur_attr, cur_pre, cur_post, _ = entry.values()

                        if '|' in cur_attr:
                            counter_attr_cluster += 1
                        if '|' in cur_pre:
                            counter_pre_cluster += 1
                        if '|' in cur_post:
                            counter_post_cluster += 1

                        counter_attr_all += 1
                        counter_pre_all += 1
                        counter_post_all += 1


    print(f'{counter_entity_cluster}/{counter_entity_all} states have clustered entity  \
            ({counter_entity_cluster/counter_entity_all*100:.2f}%)')

    print(f'{counter_attr_cluster}/{counter_attr_all} states have clustered attribute  \
            ({counter_attr_cluster/counter_attr_all*100:.2f}%)')
    
    print(f'{counter_pre_cluster}/{counter_pre_all} states have clustered precondition  \
            ({counter_pre_cluster/counter_pre_all*100:.2f}%)')

    print(f'{counter_post_cluster}/{counter_post_all} states have clustered postcondition  \
            ({counter_post_cluster/counter_post_all*100:.2f}%)')


if __name__ == '__main__':
    main()

