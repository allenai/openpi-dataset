'''
script to convert procedure to attribute only
'''
import os
import json 
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the data'
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = json.load(open(args.data_path, 'r'))

    out_data = {}
    for proc_id, proc_content in data.items():
        cur_goal, cur_steps = proc_content['goal'], proc_content['steps']
        cur_states = proc_content['states']

        entity_attr_dict = {}
        for entity_content in cur_states:
            cur_entity = entity_content['entity']
            cur_ans = entity_content['answers']
            cur_attr_lst = []

            for step_id, step_content in cur_ans.items():
                if step_content:
                    for step_dict in step_content:
                        cur_attr = step_dict['attribute']
                        if cur_attr not in cur_attr_lst:
                            cur_attr_lst.append(cur_attr)

            entity_attr_dict[cur_entity] = cur_attr_lst
        
        out_data[proc_id] = {
            'goal': cur_goal,
            'steps': cur_steps,
            'entity_attr_dict': entity_attr_dict
        }
    
    fname = args.data_path.split('/')[-1].split('.')[0]
    fpath = '/'.join(args.data_path.split('/')[:-1])
    with open(os.path.join(fpath, f'{fname}_attr.json'), 'w') as f:
        json.dump(out_data, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()