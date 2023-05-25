import os 
import json
import argparse
import pandas as pd
from glob import glob
from copy import deepcopy

def load_json(path: str):
    '''Load json file'''
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder', type=str, required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_paths = glob(args.data_folder + '/*.json')
    data_paths = [item for item in data_paths if 'parsed' in item]
    
    narrative_template= lambda a, e, b, p: f"The ({a}) of ({e}) is ({b}) before and ({p}) afterwards."
    narrative_lst = [list(item.keys())[0] for item in load_json(data_paths[0])]
    
    # check if there are repetitive entries
    assert len(narrative_lst) == len(set(narrative_lst))

    out_dict = {
        'narratives': narrative_lst,
    }
    entity_dict, attr_dict, pre_dict, post_dict = deepcopy(out_dict), deepcopy(out_dict), deepcopy(out_dict), deepcopy(out_dict)

    for data_path in data_paths:
        cur_key = ''.join(data_path.split('/')[-1].split('-')[2:]).replace('parsed.json', '')
        cur_data = load_json(data_path)
        temp_out_dict = {k: [] for k in narrative_lst}
        temp_entity_dict, temp_attr_dict, temp_pre_dict, temp_post_dict = deepcopy(temp_out_dict), deepcopy(temp_out_dict), deepcopy(temp_out_dict), deepcopy(temp_out_dict)
        for entry_dict in cur_data:
            for key, value in entry_dict.items():
                if key not in out_dict['narratives']:
                    raise ValueError('narrative not found')

                attr, ent, pre, post = value.values()
                cur_narrative = narrative_template(attr, ent, pre, post)

                temp_out_dict[key].append(cur_narrative)
                temp_entity_dict[key].append(ent)
                temp_attr_dict[key].append(attr)
                temp_pre_dict[key].append(pre)
                temp_post_dict[key].append(post)

        cur_lst = list(temp_out_dict.values())
        out_dict[cur_key] = cur_lst
        entity_dict[cur_key] = list(temp_entity_dict.values())
        attr_dict[cur_key] = list(temp_attr_dict.values())
        pre_dict[cur_key] = list(temp_pre_dict.values())
        post_dict[cur_key] = list(temp_post_dict.values())
    
    out_df = pd.DataFrame(out_dict)
    entity_df = pd.DataFrame(entity_dict)
    attr_df = pd.DataFrame(attr_dict)
    pre_df = pd.DataFrame(pre_dict)
    post_df = pd.DataFrame(post_dict)

    out_df.to_csv(args.data_folder + '/csv/combined_results.csv', index=False)
    entity_df.to_csv(args.data_folder + '/csv/entity_results.csv', index=False)
    attr_df.to_csv(args.data_folder + '/csv/attr_results.csv', index=False)
    pre_df.to_csv(args.data_folder + '/csv/precondition_results.csv', index=False)
    post_df.to_csv(args.data_folder + '/csv/postcondition_results.csv', index=False)
            
                
if __name__ == '__main__':
    main()