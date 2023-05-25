import os
import json
import pickle
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the clustering dataset'
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


def filter_entity_lst(entity_lst: list) -> list:
    '''Filter the entity list. Remove the 'the ' in the beginning of the entity name.'''
    return [item[4:] if item[:4] == 'the ' else item for item in entity_lst]


def main():
    args = get_args()
    original_data = load_json(args.data_path)

    entity_lst = []
    for key, val in original_data.items():
        gt = val['entity_states']
        entity_lst += [item_dict['entity'].strip() for item_dict in gt]
    
    unique_entity_lst = filter_entity_lst(list(set(entity_lst)))

    print(f'There are {len(entity_lst)} unique entries in total.')
    print(f'There are {len(entity_lst) - len(unique_entity_lst)} repetitive entries.')
    
    # initialize the entity dict
    # key are canonical names, values are other names of the entity
    entity_dict = [{'canonical_name': entity, 'other_names': []} for entity in unique_entity_lst]
    entity_to_idx = {entity: idx for idx, entity in enumerate(unique_entity_lst)}

    for val in original_data.values():
        generated = val['gpt3_output']
        for entity, content in generated.items():
            gt_entity = filter_entity_lst([entity])[0]
            cur_idx = entity_to_idx[gt_entity]
            cur_entity = [gt_entity]
            for item_dict in content:
                if item_dict['structured']:
                    item_entity_lst = [d['entity'] for d in item_dict['structured']]
                    cur_entity += list(set(filter_entity_lst(item_entity_lst)))
            
            cur_entity = list(set(cur_entity))
            entity_dict[cur_idx]['other_names'] += cur_entity
    
    rm_idx = []
    for i, content_dict in enumerate(tqdm(entity_dict)):
        for j, other_dict in enumerate(entity_dict):
            if i != j:
                cur_id = other_dict['canonical_name']
                ind = sum([item == cur_id for item in content_dict['other_names']])
                if cur_id == content_dict['canonical_name'] or ind > 0:
                    content_dict['other_names'] += other_dict['other_names'] + [other_dict['canonical_name']]
                    if j not in rm_idx:
                        rm_idx.append(j)
                    break

        content_dict['other_names'] = list(set(content_dict['other_names']))
    
    entity_dict = [item for (i, item) in enumerate(entity_dict) if i not in rm_idx]

    # print(entity_dict)
    with open(args.out_path, 'w') as f:
        json.dump(entity_dict, f, indent=4)
    f.close()

    with open(args.out_path.replace('.json', '.pkl'), 'wb') as f:
        pickle.dump(entity_to_idx, f)
    f.close()


if __name__ == '__main__':
    main()