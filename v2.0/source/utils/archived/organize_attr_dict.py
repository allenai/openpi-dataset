import re
import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the clustering dataset'
    )
    # parser.add_argument(
    #     '--out_path', type=str, required=True, help='path to the output file'
    # )
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

    attr_list = []
    for key, val in original_data.items():
        gt = val['entity_states']
        components = [item_dict['components'] for item_dict in gt]

        for component_dicts in components:
            attr_list += [item['attribute'] for item in component_dicts]
    
    unique_attr_list = set(list(attr_list))
    print(f'Number of unique attributes: {len(unique_attr_list)}')
    print(f'Number of duplicated attributes: {len(attr_list) - len(unique_attr_list)}')

    openpi_template = re.compile('The (.*) of (.*) is (.*) before and (.*) afterwards.')

    attr_to_idx = {attr: idx for idx, attr in enumerate(unique_attr_list)}
    attr_out_dict = [{'canonical_name': attr, 'other_names': []} for attr in unique_attr_list]
    for val in original_data.values():
        generated = val['gpt3_output']

        for component_lst in generated.values():
            for narrative in component_lst:
                if narrative['structured']:
                    gt_attr = openpi_template.findall(narrative['input'])[0][0]
                    try:
                        cur_idx = attr_to_idx[gt_attr]
                    except:
                        gt_attr = gt_attr.split(' of ')[0].strip()
                        cur_idx = attr_to_idx[gt_attr]
                    
                    gen_attr = [item['attribute'] for item in narrative['structured']]
                    gen_attr = [item for item in gen_attr if item != gt_attr]
                    attr_out_dict[cur_idx]['other_names'] += gen_attr
    
    for idx, component_dict in enumerate(attr_out_dict):
        attr_out_dict[idx]['other_names'] = list(set(component_dict['other_names']))

    with open('../../data/gpt-gen-data/dev_attr_grouped.json', 'w') as f:
        json.dump(attr_out_dict, f, indent=4)
    f.close()
    

    




if __name__ == '__main__':
    main()