import json

from utils import load_json, load_txt


def main() -> None:
    template = load_txt('./assets/templates/attr_chat_template.txt')
    original_example_data = load_json('../../data/example_data_attr.json') 
    annotated_example_data = load_json('../../data/example_data_annotated_attr.json')

    for format in ['vanilla', 'concise']:
        out_dict = {}
        for proc_id, content in original_example_data.items():
            cur_goal = content['goal']
            cur_steps = content['steps']
            gt_content = annotated_example_data[proc_id]['entity_attr_dict']

            cur_step_narratives = []
            for i, step in enumerate(cur_steps):
                if i == 0:
                    cur_step_narratives.append(f'First, {step.lower()}')
                else:
                    cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, {step.lower()}')

            cur_steps = cur_step_narratives[-1]

            cur_entity_attr_dict = content['entity_attr_dict']
            
            for entity, attr_lst in cur_entity_attr_dict.items():
                cur_entity = entity.split(' | ')[0]
                attr_lst = [item.split(' | ') for item in attr_lst]
                attr_lst = [lst for sublst in attr_lst for lst in sublst]

                gt_entity_lst = [item.split(' | ') for item in list(gt_content.keys())]
                for ent_lst in gt_entity_lst:
                    if cur_entity in ent_lst:
                        gt_attr_lst = gt_content[' | '.join(ent_lst)]
                        break

                if len(attr_lst) > 1:
                    for i in range(len(attr_lst)):
                        for j in range(i+1, len(attr_lst)):
                            attr1, attr2 = attr_lst[i], attr_lst[j]
                            ind = 0
                            for gt_attr in gt_attr_lst:
                                if attr1 in gt_attr and attr2 in gt_attr:
                                    ind = 1
                                    break
                            
                            cur_goal = cur_goal.replace('"', "'")
                            cur_steps = cur_steps.replace('"', "'")
                            attr1, attr2 = attr1.replace('"', "'"), attr2.replace('"', "'")

                            if format == 'vanilla':
                                pass
                            elif format == 'concise':
                                attr1 = f'{attr1} of {cur_entity}'
                                attr2 = f'{attr2} of {cur_entity}'

                            cur_template = template.replace('{goal}', cur_goal)  \
                                                .replace('{steps}', cur_steps)  \
                                                .replace('{attributes}', f'({attr1}, {attr2})')
                            
                            if ind:
                                cur_template = cur_template.replace('{annotation}', 'Yes')
                            else:
                                cur_template = cur_template.replace('{annotation}', 'No')
                                
                            out_dict[f'{proc_id}_{cur_entity}_{i}_{j}'] = cur_template
        
        with open(f'./assets/examples/attribute_chat_{format}_example.json', 'w') as f:
            json.dump(out_dict, f, indent=4)
        f.close()


if __name__ == '__main__':
    main()


