import json
import pickle
import argparse


def load_json(data_path: str) -> dict:
    '''Load json file.'''
    with open(data_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def save_json(data: dict, data_path: str) -> None:
    '''Save json file.'''
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)
    f.close()


def load_pickle(data_path: str) -> dict:
    '''Load pickle file.'''
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


def get_args():
    '''Get arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the openpi dataset'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to the output file'
    )
    parser.add_argument(
        '--entity_count_path', type=str, required=True, help='path to the entity count statistics'
    )
    parser.add_argument(
        '--attr_count_path', type=str, required=True, help='path to the attribute count statistics'
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = load_json(args.data_path)

    # organize info on entity and attribute
    attr_count = load_pickle(args.attr_count_path)
    entity_count = load_pickle(args.entity_count_path)
    attr_global_count = [lst for sublst in attr_count for lst in sublst]
    entity_global_count = [lst for sublst in entity_count for lst in sublst]

    attr_global_count_keys = [list(item.keys()) for item in attr_global_count]
    attr_global_count_keys = [lst for sublst in attr_global_count_keys for lst in sublst]
    entity_global_count_keys = [list(item.keys()) for item in entity_global_count]
    entity_global_count_keys = [lst for sublst in entity_global_count_keys for lst in sublst]

    toy_example = data['1']

    cur_entities = toy_example['entities']
    cur_attributes = toy_example['attributes']

    # get global annotation count (across all steps)
    for idx, entity_dict in enumerate(cur_entities):
        cur_count = []

        if (cur_entity := entity_dict['canonical_name']) in entity_global_count_keys:
            cur_count += [item.get(cur_entity, []) for item in entity_global_count]
        
        for cur_entity in entity_dict['other_names']:
            if cur_entity in entity_global_count_keys:
                cur_count += [item.get(cur_entity, []) for item in entity_global_count]
        
        cur_count = list(set([lst for sublst in cur_count for lst in sublst]))
        toy_example['entities'][idx]['people_annotated'] = cur_count
        toy_example['entities'][idx]['num_people_annotated'] = len(cur_count)
    
    for idx, attr_dict in enumerate(cur_attributes):
        cur_count = []

        for entity_dict in cur_entities:
            # first check canonical names
            cur_tup = tuple((entity_dict['canonical_name'], attr_dict['canonical_name']))
            if cur_tup in attr_global_count_keys:
                cur_count += [item.get(cur_tup, []) for item in attr_global_count]
            
            for cur_entity in entity_dict['other_names']:
                cur_tup = tuple((cur_entity, attr_dict['canonical_name']))
                if cur_tup in attr_global_count_keys:
                    cur_count += [item.get(cur_tup, []) for item in attr_global_count]
            
            # then check other names
            for cur_attr in attr_dict['other_names']:
                cur_tup = tuple((entity_dict['canonical_name'], cur_attr))
                if cur_tup in attr_global_count_keys:
                    cur_count += [item.get(cur_tup, []) for item in attr_global_count]

                for cur_entity in entity_dict['other_names']:
                    cur_tup = tuple((cur_entity, cur_attr))
                    if cur_tup in attr_global_count_keys:
                        cur_count += [item.get(cur_tup, []) for item in attr_global_count]
            
        cur_count = list(set([lst for sublst in cur_count for lst in sublst]))
        toy_example['attributes'][idx]['people_annotated'] = cur_count
        toy_example['attributes'][idx]['num_people_annotated'] = len(cur_count)
    
    save_json(toy_example, args.out_path)


if __name__ == '__main__':
    main()