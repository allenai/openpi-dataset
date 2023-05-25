import os
import ast
import json


def main():
    data = json.load(open('../../data/data_in_new_format/dev-data-reformatted-v4-overgenerated.json', 'r'))

    out_file = {}
    for proc_id, proc_content in data.items():
        if int(proc_id) > 20:
            break
        cur_clusters = proc_content['clusters']
        cur_goal = proc_content['goal']
        cur_steps = proc_content['steps']

        cur_steps = '\n'.join([f'[Step{i+1}] {content}' for (i, content) in enumerate(cur_steps)])

        temp_dict = { 
            'goal': cur_goal, 
            'steps': cur_steps, 
            'annotation': {} 
        }

        for entity_id, entity_cluster in cur_clusters.items():
            original_cluster = entity_cluster['entity_cluster']
            cur_og = entity_cluster['entity_overgenerate']

            temp_dict['annotation'][entity_id] = {
                'original_cluster': original_cluster,
                'overgenerate_cluster': cur_og,
            }

            display_og = '\n'.join([f'#{i} {content}' for (i, content) in enumerate(cur_og)])

            os.system('clear')
            print('----------------------------------------------------')
            print(f'Progress: {proc_id}/{len(data)}')
            print('----------------------------------------------------')
            print(f'Goal: {cur_goal}')
            print('----------------------------------------------------')
            print(f'Steps:\n{cur_steps}')
            print('----------------------------------------------------')
            print('----------------------------------------------------')
            print(f'Original Entities: {original_cluster}')
            print(f'Over Generated Entities:\n\n{display_og}')
            print('----------------------------------------------------\n')

            bad_og = input('Bad Over Generated Entities (Index): ')
            bad_og = ast.literal_eval('[' + ','.join(bad_og.split(',')) + ']')
            bad_og = [int(item) for item in bad_og]
            bad_og = [cur_og[idx] for idx in bad_og]

            temp_dict['annotation'][entity_id]['bad_overgenerate'] = bad_og
            

        out_file[proc_id] = temp_dict

    with open('../../data/over_generate_relevance_annotation/v4-entity-annotation.json', 'w') as f:
        json.dump(out_file, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()
