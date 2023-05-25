''' 
script to parse the gpt3 generated openpi descriptions
'''

import re
import json
import argparse


def get_args():
    '''Get arguments from command line.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True
    )
    parser.add_argument(
        '--out_path', type=str, required=True
    )
    return parser.parse_args()


def load_json(data_path: str) -> dict:
    '''Load json file.'''

    with open(data_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def save_json(out_path: str, out_dict: dict) -> None:
    '''Save json file.'''

    with open(out_path, 'w') as f:
        json.dump(out_dict, f, indent=4)
    f.close()


def get_template() -> str:
    '''Get the template of the narrative.'''
    return re.compile('The (.*) of (.*) is (.*) before and (.*) afterwards')


def remove_bar(s: str) -> str:
    '''Remove the leading bar in the string.'''
    if s[:3] == ' | ':
        s = s[3:]
    return s


def main():
    args = get_args()
    data = load_json(args.data_path)
    template = get_template()

    out_dict = {}
    for key, value in data.items():
        cur_dict = {
            'goal': value['goal'],
            'steps': value['steps'],
            'entity_state': [{'entity': item['entity'], 'openpi_narrative': item['openpi_narrative']} for item in value['entity_states']]
        }

        narrative_all = [d['openpi_narrative'] for d in cur_dict['entity_state']]
        narrative_all = [lst for sublst in narrative_all for lst in sublst]
        narrative_all = list(set(narrative_all))

        narrative_dict = [{k: []} for k in narrative_all]
        
        cur_entity_state = value['gpt3_output']
        for entity, state_lst in cur_entity_state.items():
            for state_dict in state_lst:
                narrative = state_dict['input']
                components = state_dict['structured']
                cur_state_dict = {
                    'attribute': ' | '.join(list(set([ele['attribute'] for ele in components]))),
                    'entity': ' | '.join(list(set([ele['entity'] for ele in components]))),
                    'before': ' | '.join(list(set([ele['pre-state'] for ele in components]))),
                    'after': ' | '.join(list(set([ele['post-state'] for ele in components]))),
                }
                narrative_dict[narrative_all.index(narrative)][narrative].append(cur_state_dict)
    
    # another round of combining result
    visited_key = []
    for idx, content in enumerate(narrative_dict):
        for key, component_lst in content.items():
            if key in visited_key:
                continue
            else:
                visited_key.append(key)

            gt_attr, gt_entity, gt_before, gt_after = re.findall(template, key)[0]

            if len(component_lst) > 1:
                attr, entity, before, after = [gt_attr.strip()], [gt_entity.strip()], [gt_before.strip()], [gt_after.strip()]

                for component_dict in component_lst:
                    attr += [component_dict['attribute']]
                    entity += [component_dict['entity']]
                    before += [component_dict['before']]
                    after += [component_dict['after']]
                
                attr = ' | '.join([ele.strip() for ele in list(set(attr))])
                entity = ' | '.join([ele.strip() for ele in list(set(entity))])
                before = ' | '.join([ele.strip() for ele in list(set(before))])
                after = ' | '.join([ele.strip() for ele in list(set(after))])

                attr, entity, before, after = remove_bar(attr), remove_bar(entity), remove_bar(before), remove_bar(after)

                new_component_dict = {
                    'attribute': attr.strip(),
                    'entity': entity.strip(),
                    'before': before.strip(),
                    'after': after.strip()
                }

                narrative_dict[idx][key] = new_component_dict
            else:
                try:
                    narrative_dict[idx][key] = component_lst[0]
                except:
                    narrative_dict[idx][key] = []
    
    # save the result
    save_json(args.out_path, narrative_dict)


if __name__ == '__main__':
    main()
