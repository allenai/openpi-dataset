import ast
import json
import argparse

from utils import load_json, load_txt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--format', type=str, default='vanilla', help='format of the output, choose from ["vanilla", "concise"]'
    )
    return parser.parse_args()


def fill_template(template: str, gt_entity_attr: list, cur_goal: str, cur_steps: str, entity_attr: list, entity: str):
    entity_attr = [item.split(' | ') for item in entity_attr]
    entity_attr = [lst for sublst in entity_attr for lst in sublst]
    entity_attr = [item.strip() for item in entity_attr]

    entity = entity.split('|')[0].strip()

    
    if args.format == 'concise':
        entity_attr = [f'{item} of {entity}' for item in entity_attr]

        clustered_attr = ''
        for entry in gt_entity_attr:
            entry_lst = [item.strip() for item in entry.split(' | ')]
            entry_lst = [f'{item} of {entity}' for item in entry_lst]
            clustered_attr += f"\\n- {', '.join(entry_lst)}"

    else:
        clustered_attr = ''
        for entry in gt_entity_attr:
            clustered_attr += f"\\n- {', '.join([item.strip() for item in entry.split(' | ')])}"

    cur_template = template.replace('{goal}', cur_goal)  \
                            .replace('{steps}', cur_steps)  \
                            .replace('{attributes}', str(entity_attr).replace('[', '').replace(']', ''))  \
                            .replace('{grouped_attr}', clustered_attr)
    return cur_template


def main() -> str:
    global args
    args = get_args()
    original_example_data = load_json('../../data/example_data_attr.json') 
    annotated_example_data = load_json('../../data/example_data_annotated_attr.json')
    template = load_txt('./assets/templates/attr_chat_template_v2.txt')

    out_dict = {}
    for proc_id, content in original_example_data.items():
        cur_goal = content['goal']
        cur_steps = content['steps']

        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, I {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, I {step.lower()}')

        cur_steps = cur_step_narratives[-1]

        cur_entity_attr_dict = content['entity_attr_dict']
        cur_gt_entity_attr_dict = annotated_example_data[proc_id]['entity_attr_dict']

        gt_entity_attr_keys = list(cur_gt_entity_attr_dict.keys())

        for idx, (entity_id, entity_attr) in enumerate(cur_entity_attr_dict.items()):
            
            gt_entity_attr = cur_gt_entity_attr_dict[gt_entity_attr_keys[idx]]

            if len(gt_entity_attr) > 1:
                cur_template = fill_template(template, gt_entity_attr, cur_goal, cur_steps, entity_attr, entity_id)
                out_dict[f'{proc_id}_{entity_id}'] = ast.literal_eval(cur_template)
                out_dict[f'{proc_id}_{entity_id}'] = ast.literal_eval(cur_template)
    
    with open(f'./assets/examples/attribute_chat_{args.format}_example.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()



