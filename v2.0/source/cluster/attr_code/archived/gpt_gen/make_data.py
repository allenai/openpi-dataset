import argparse
import numpy as np

from utils import load_json, save_json, load_template


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the openpi dataset'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='specify the location to save the resulting file'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='seed for numpy'
    )
    parser.add_argument(
        '--num_samples', type=int, default=-1, help='number of samples to get from openpi dataset'
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = load_json(args.data_path)

    openpi_template = lambda attr, ent, pre, post: f'The {attr} of {ent} is {pre} before and {post} afterwards.'

    if args.num_samples == -1:
        data_keys_selected = list(data.keys())
    else:
        np.random.seed(args.seed)
        data_keys_selected = np.random.choice(list(data.keys()), args.num_samples, replace=False)
    
    sampled_data = {k: v for (k, v) in data.items() if k in data_keys_selected}
    
    if args.num_samples != -1:
        assert len(sampled_data) == args.num_samples
    
    out_dict = {}
    for index, entry in sampled_data.items():
        goal, steps, _, _, states = entry.values()
        entities = []

        for state_dict in states:
            cur_entity = state_dict['entity']
            cur_entity_lst = list(set([ele.strip() for ele in cur_entity.split('|')]))

            # take only the first entity in the cluster
            cur_entity_lst = [cur_entity_lst[0]]

            for entity in cur_entity_lst:
                if entity not in entities:
                    entities.append(entity)

        temp_dict = {
            'goal': goal,
            'steps': steps,
            'entity_states': [],
        }

        for entity in entities:
            for state_dict in states:
                cur_entity = state_dict['entity']
                cur_entity_lst = list(set([ele.strip() for ele in cur_entity.split('|')]))

                cur_entity_dict = {}
                if entity in cur_entity_lst:    
                    temp_entity_lst = []
                    cur_answers = state_dict['answers']
                    temp_entity_dict = {
                        'entity': entity,
                        'openpi_narrative': [],
                        'components': []
                    }

                    for step_content_lst in cur_answers.values():
                        for step_content in step_content_lst:
                            attr, before, after, _ = step_content.values()
                            attr_lst = [ele.strip() for ele in attr.split('|')]
                            before = before.split('|')[0].strip()
                            after = after.split('|')[0].strip()

                            for attr in attr_lst:
                                temp_entity_dict['openpi_narrative'].append(openpi_template(attr, entity, before, after))
                                cur_components = {
                                    'attribute': attr,
                                    'precondition': before,
                                    'postcondition': after
                                }
                                temp_entity_dict['components'].append(cur_components)

                    temp_dict['entity_states'].append(temp_entity_dict)                             
        
            out_dict[index] = temp_dict

        # save the output dict
        save_json(args.out_path, out_dict)


if __name__ == '__main__':
    main()

