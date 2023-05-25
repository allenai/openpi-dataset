import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cluster_path', type=str, required=True, help='path to the cluster file'
    )
    return parser.parse_args()


def remove_determinant(word: str) -> str:
    word = word.strip()
    if word.startswith('the '):
        return word[4:]
    elif word.startswith('a '):
        return word[2:]
    elif word.startswith('an '):
        return word[3:]
    else:
        return word


def plural_to_singular(word: str) -> str:
    word = word.strip()
    if not word.endswith('ss') and len(word) > 4:
        if word.endswith('ies'):
            return word[:-3] + 'y'
        if word.endswith('s'):
            return word[:-1]
        else:
            return word
    else:
        return word


def main():
    args = get_args()
    original_data = json.load(open('../../data/dev-ranked.json', 'r'))
    entity_cluster = json.load(open(args.cluster_path, 'r'))

    for proc_id, proc_content in original_data.items():
        cur_entity_cluster = entity_cluster[proc_id]
        cur_states = proc_content['states']

        # for key, val in cur_entity_cluster.items():
        #     cur_entity_cluster[key].append(key)

        original_data[proc_id]['entity_clusters'] = cur_entity_cluster

        for entity_id, entity_content in enumerate(cur_states):
            cur_original_entity =  entity_content['entity']
            cur_entity_lst = [item.strip() for item in entity_content['entity'].split('|')]

            for cur_entity in cur_entity_lst:
                cur_entity_resolved = 0
                cur_entity_no_determinant = remove_determinant(cur_entity)
                cur_entity_singular = plural_to_singular(cur_entity_no_determinant)

                for key, val in cur_entity_cluster.items():
                    if cur_entity == key or cur_entity_no_determinant == key or cur_entity_singular == key:
                        original_data[proc_id]['states'][entity_id]['entity'] = key
                        original_data[proc_id]['states'][entity_id]['entity_annotation'] = [cur_original_entity]
                        cur_entity_resolved = 1
                        break
                    elif cur_entity in val or cur_entity_no_determinant in val or cur_entity_singular in val:
                        original_data[proc_id]['states'][entity_id]['entity'] = key
                        original_data[proc_id]['states'][entity_id]['entity_annotation'] = [cur_original_entity]
                        cur_entity_resolved = 1
                        break
                
                assert cur_entity_resolved == 1, f'entity {cur_entity} not resolved'
        
    # for states in a procedure, combine annotations with the same cannonical entity
    for proc_id, proc_content in original_data.items():
        cur_state_lst = proc_content['states']
        visited_entities = []
        remove_idx = []
        for idx, entity_entry in enumerate(cur_state_lst):
            cur_entity = entity_entry['entity']
            if cur_entity not in visited_entities:
                visited_entities.append(cur_entity)
            else:
                search_lst = cur_state_lst[:idx]
                remove_idx.append(idx)
                for j, entity_dict in enumerate(search_lst):
                    if entity_dict['entity'] == cur_entity:
                        for step_id, step_content in entity_dict['answers'].items():
                            cur_existing_attr = [item['attribute'] for item in entity_dict['answers'][step_id]]
                            cur_additional_info = [item for item in step_content if item['attribute'] not in cur_existing_attr]
                            original_data[proc_id]['states'][j]['answers'][step_id] += cur_additional_info
                        
                        if entity_entry['entity_annotation'] not in entity_dict['entity_annotation']:
                            original_data[proc_id]['states'][j]['entity_annotation'] += entity_entry['entity_annotation']
                        break
        
        original_data[proc_id]['states'] = [item for idx, item in enumerate(original_data[proc_id]['states']) if idx not in remove_idx]
    
    # add original entity to entity cluster if it is ignored by the model
    for proc_id, proc_content in original_data.items():

        # cluster from model
        cur_cluster_dict = proc_content['entity_clusters']

        cur_model_entity_lst = list(cur_cluster_dict.keys())
        for val in cur_cluster_dict.values():
            cur_model_entity_lst.extend(val)
        
        # entities from annotator
        cur_entity_lst = [item['entity_annotation'] for item in proc_content['states']]
        cur_entity_lst = [lst for sublst in cur_entity_lst for lst in sublst]
        cur_entity_lst = [item.split(' | ') for item in cur_entity_lst]
        cur_entity_lst = [lst for sublst in cur_entity_lst for lst in sublst]
        cur_entity_lst = [item.strip() for item in cur_entity_lst]

        for entity in cur_entity_lst:
            if entity not in cur_model_entity_lst:
                # add the entity to the cluster
                nondet_entity = remove_determinant(entity)
                singular_entity = plural_to_singular(nondet_entity)

                for entity_key, cluster in proc_content['entity_clusters'].items():

                    if nondet_entity == entity_key or nondet_entity in cluster:
                        original_data[proc_id]['entity_clusters'][entity_key].append(entity)

                    elif singular_entity == entity_key or singular_entity in cluster:
                        original_data[proc_id]['entity_clusters'][entity_key].append(entity)
        
        # sanity check
        cur_cluster_dict = proc_content['entity_clusters']
        cur_model_entity_lst = list(cur_cluster_dict.keys())
        for val in cur_cluster_dict.values():
            cur_model_entity_lst.extend(val)

        for entity in cur_entity_lst:
            if entity not in cur_model_entity_lst:
                raise Exception(f'entity {entity} not in cluster')


    with open('../../data/data_in_new_format/dev-data-reformatted-v3.json', 'w') as f:
        json.dump(original_data, f, indent=4)
    f.close()



if __name__ == '__main__':
    main()