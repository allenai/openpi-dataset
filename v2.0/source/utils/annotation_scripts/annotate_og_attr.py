import os
import ast
import json


def main():
    data = json.load(open('../../../data/data_in_new_format/dev-data-reformatted-v4-overgenerated-unique.json', 'r'))

    out_file = {}
    for proc_id, proc_content in data.items():
        if int(proc_id) > 20:
            break
        cur_clusters = proc_content['clusters']
        cur_goal = proc_content['goal']
        cur_steps = proc_content['steps']
        cur_states = proc_content['states']

        cur_steps = '\n'.join([f'[Step{i+1}] {content}' for (i, content) in enumerate(cur_steps)])

        temp_dict = { 
            'goal': cur_goal, 
            'steps': cur_steps, 
            'annotation': {} 
        }

        for entity_id, entity_cluster in cur_clusters.items():
            original_cluster = entity_cluster['entity_cluster']

            entity_state = [state for state in cur_states if state['entity'] == entity_id][0]
            cur_answers = entity_state['answers']

            temp_dict['annotation'][entity_id] = {
                'original_entity': original_cluster,
                'attribute_annotation': [],
            }

            attr_clusters = entity_cluster['attribute_cluster']

            for attr_id, attr_cluster in attr_clusters.items():
                original_attr = attr_cluster['attribute_cluster']
                og_attr = attr_cluster['attribute_overgenerate']

                # get pre and post condition for current attribute
                cur_pre, cur_post = '', ''
                for annotation_lst in cur_answers.values():
                    for annotation in annotation_lst:
                        if annotation['attribute'] == attr_id:
                            cur_pre = annotation['before']
                            cur_post = annotation['after']
                            break
                    if cur_pre and cur_post:
                        break

                if cur_pre != '' and cur_post != '':

                    og_display = '\n'.join([f'#{i}: {attr}' for (i, attr) in enumerate(og_attr)])

                    os.system('clear') 
                    print('----------------------------------------------------')
                    print(f'Progress: {proc_id}/{len(data)}')
                    print('----------------------------------------------------')
                    print(f'Goal: {cur_goal}')
                    print('----------------------------------------------------')
                    print(f'Steps:\n{cur_steps}')
                    print('----------------------------------------------------')
                    print('----------------------------------------------------')
                    print(f'Current Entities: {original_cluster}')
                    print('----------------------------------------------------')
                    print('----------------------------------------------------')
                    print(f'Before: {cur_pre}')
                    print(f'After: {cur_post}')
                    print('----------------------------------------------------')
                    print(f'Current Attribute: {original_attr}')
                    print('----------------------------------------------------')
                    print(f'Current Attribute:\n{og_display}')
                    print('----------------------------------------------------\n')

                    bad_og = input('Bad Over Generated Attribute (Index): ')
                    bad_og = ast.literal_eval('[' + ','.join(bad_og.split(',')) + ']')
                    bad_og = [int(item) for item in bad_og]
                    bad_og = [og_attr[idx] for idx in bad_og]

                    temp_dict['annotation'][entity_id]['attribute_annotation'].append(
                        {
                            'original_attribute': original_attr,
                            'attribute_overgenerate': og_attr,
                            'bad_overgenerate': bad_og,
                        }
                    )

        out_file[proc_id] = temp_dict

    with open('../../../data/over_generate_relevance_annotation/v4-attribute-annotation.json', 'w') as f:
        json.dump(out_file, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()
