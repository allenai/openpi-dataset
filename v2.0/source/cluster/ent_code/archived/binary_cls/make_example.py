import json

from utils import load_json, load_txt


def main() -> None:
    template = load_txt('./assets/templates/entity_chat_template.txt')
    original_example_data = load_json('../../data/example_data_entity.json') 
    annotated_example_data = load_json('../../data/example_data_annotated_entity.json')

    out_dict = {}
    for proc_id, content in original_example_data.items():
        cur_goal = content['goal']
        cur_steps = content['steps']

        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, {step.lower()}')

        cur_steps = cur_step_narratives[-1]

        cur_entities = content['entities_flattened']
        cur_gt_entities = annotated_example_data[proc_id]['entities']

        for i in range(len(cur_entities)):
            for j in range(len(cur_entities)):
                if i != j:
                    ent1, ent2 = cur_entities[i], cur_entities[j]
                    ind = 0
                    for lst in cur_gt_entities:
                        if ent1 in lst and ent2 in lst:
                            ind = 1
                            break
                    
                    cur_goal = cur_goal.replace('"', "'")
                    cur_steps = cur_steps.replace('"', "'")
                    ent1, ent2 = ent1.replace('"', "'"), ent2.replace('"', "'")

                    cur_template = template.replace('{goal}', cur_goal)  \
                                           .replace('{steps}', cur_steps)  \
                                           .replace('{entities}', f'({ent1}, {ent2})')
                    
                    if ind == 1:
                        cur_template = cur_template.replace('{annotation}', 'Yes')
                    else:
                        cur_template = cur_template.replace('{annotation}', 'No')
                    
                    out_dict[f'{proc_id}_{i}_{j}'] = cur_template
    
    with open('./assets/examples/entity_chat_example.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()


