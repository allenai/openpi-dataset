import ast
import json

from utils import load_json, load_txt


def main() -> str:
    original_example_data = load_json('../../data/example_data_entity.json') 
    annotated_example_data = load_json('../../data/example_data_annotated_entity.json')
    template = load_txt('./assets/templates/entity_chat_template_v2.txt')

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

        cur_entities = content['entities_flattened']
        cur_gt_entities = '\\n- '.join([str(item) for item in annotated_example_data[proc_id]['entities']])

        cur_goal = cur_goal.replace('"', "'")
        cur_steps = cur_steps.replace('"', "'")

        cur_template = template.replace('{goal}', cur_goal)  \
                                .replace('{steps}', cur_steps)  \
                                .replace('{entities}', str(cur_entities))  \
                                .replace('{grouped_entities}', '\\n- ' + cur_gt_entities)
        
        cur_template = [ast.literal_eval(item) for item in cur_template.strip().split('\n')]
        out_dict[proc_id] = cur_template
        
    
    with open('./assets/examples/entity_chat_example.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()



