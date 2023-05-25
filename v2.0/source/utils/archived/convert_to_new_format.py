import os
import re 
import ast
import json
import pickle
import argparse


def get_args():
    '''Get arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the openpi dataset'
    )
    parser.add_argument(
        '--entity_dict_path', type=str, required=True, help='path to the entity dictionary'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to the output file'
    )
    return parser.parse_args()


def load_json(data_path: str) -> dict:
    '''Load json file.'''
    with open(data_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def save_json(data: dict, data_path: str) -> None:
    '''Save json file.'''
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)
    f.close()


def load_pickle(data_path: str) -> dict:
    '''Load pickle file.'''
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


def get_canonical_name(name: str) -> str:
    '''Get canonical name from the name in the dataset. Use the first name in the list.'''
    return name[0]


def clean_entity_name(entity: str) -> str:
    '''Clean entity name. Remove 'the' in the beginning.'''
    if entity.strip()[:3].lower() == 'the':
        entity = entity[4:].strip()
    return entity.strip()


def remove_determinant(entity: str) -> str:
    '''Remove the determinants in the entity name.'''
    if entity.strip()[:3].lower() == 'the':
        entity = entity[4:].strip()
    if entity.strip()[:2].lower() == 'a ':
        entity = entity[2:].strip()
    if entity.strip()[:2].lower() == 'an':
        entity = entity[2:].strip()
    return entity.strip()


def main():
    args = get_args()
    original_data = load_json(args.data_path)
    entity_dict = load_json(args.entity_dict_path)

    for key, val in original_data.items():

        cur_entity_dict = entity_dict[key]
        cur_entity_clusters = cur_entity_dict['grouped_entities']
        cur_entity_original = cur_entity_dict['original_entities']
        cur_entity_original = [item.split('|') for item in cur_entity_original]
        cur_entity_original = [lst for sublst in cur_entity_original for lst in sublst]
        cur_entity_original = [item.strip() for item in cur_entity_original]
        cur_entity_clusters_all = [item.split(',') for item in cur_entity_clusters]
        cur_entity_clusters_all = [lst for sublst in cur_entity_clusters_all for lst in sublst]
        cur_entity_clusters_all = [item.strip() for item in cur_entity_clusters_all]

        for entity in cur_entity_original:
            if entity not in cur_entity_clusters_all:
                cur_entity_clusters.append(entity)

        cur_entity_cluster_dict = {}
        visited_entities = []
        for cluster in cur_entity_clusters:
            cluster = [item.strip() for item in cluster.split(',')]
            id = get_canonical_name(cluster).strip().lower()
            if id not in visited_entities:
                visited_entities.append(id.lower())
                if len(id) >= 1 and max([len(item.split()) for item in id.split(',')]) <= 6:
                    cur_entity_cluster_dict[id] = {
                        'entity_cluster': cluster
                    }
    
        original_data[key]['clusters'] = cur_entity_cluster_dict

        cur_states = val['states']

        for idx, item_dict in enumerate(cur_states):

            # make entity clusters
            cur_entity = item_dict['entity'].lower()
            if '|' in cur_entity:
                cur_entity = cur_entity.split('|')[0]

            ind = 0
            for entity_id, val in cur_entity_cluster_dict.items():
                val = val['entity_cluster']
                if entity_id == cur_entity.strip():
                    cur_states[idx]['entity'] = entity_id
                    ind = 1
                    break

                elif cur_entity.strip() in val:
                    cur_states[idx]['entity'] = entity_id
                    ind = 1
                    break

                elif remove_determinant(cur_entity).strip() == entity_id:
                    cur_states[idx]['entity'] = entity_id
                    ind = 1
                    break

                elif remove_determinant(cur_entity).strip() in val:
                    cur_states[idx]['entity'] = entity_id
                    ind = 1
                    break
                
                else:
                    for sub_entity in cur_entity.split(','):
                        if sub_entity.lower() == entity_id:
                            cur_states[idx]['entity'] = entity_id
                            ind = 1
                            break
                        elif sub_entity.lower() in val:
                            cur_states[idx]['entity'] = entity_id
                            ind = 1
                            break

            if not ind:
                raise ValueError(f'Entity [{cur_entity}] not found in the entity clusters.')
            
            # make attribute clusters
            cur_answers = item_dict['answers']
            cur_attr = [[item['attribute'] for item in lst] for lst in cur_answers.values()]
            cur_attr = list(set([lst for sublst in cur_attr for lst in sublst]))

            attr_cluster = {}
            for attr in cur_attr:
                if ' | ' in attr:
                    attr = attr.split(' | ')
                    id = get_canonical_name(attr).strip()
                    attr_cluster[id] = attr
                else:
                    attr_cluster[attr] = [attr]

            ind = 1
            for entity, entity_cluster in original_data[key]['clusters'].items():
                cur_entity = cur_entity.strip().lower()
                entity_cluster = entity_cluster['entity_cluster']
                if entity == cur_entity:
                    original_data[key]['clusters'][entity]['attribute_cluster'] = attr_cluster
                    ind = 0
                    break
                elif cur_entity in entity_cluster:
                    original_data[key]['clusters'][entity]['attribute_cluster'] = attr_cluster
                    ind = 0
                    break
                elif [item.strip() for item in cur_entity.split(',')] == entity_cluster:
                    original_data[key]['clusters'][entity]['attribute_cluster'] = attr_cluster
                    ind = 0
                    break
                elif entity in cur_entity:
                    original_data[key]['clusters'][entity]['attribute_cluster'] = attr_cluster
                    ind = 0
                    break

            if ind:
                print(original_data[key]['clusters'])
                raise ValueError(f'Entity [{cur_entity}] not found in the entity clusters.')

            for step_num, step_content_lst in cur_answers.items():
                for j, content_dict in enumerate(step_content_lst):
                    attr = content_dict['attribute']
                    if ' | ' in attr:
                        attr = attr.split(' | ')
                        for id, attr_list in attr_cluster.items():
                            if attr == attr_list:
                                cur_answers[step_num][j]['attribute'] = id
                                break
                            elif attr == id:
                                break
                    else:
                        for id, attr_list in attr_cluster.items():
                            if attr in attr_list:
                                cur_answers[step_num][j]['attribute'] = id
                                break
                            elif attr == id:
                                break
    
    # another round of sanity checks
    for key, val in original_data.items():
        cur_state_lst = val['states']
        visited_entities, delete_idx = [], []

        for idx, state_dict in enumerate(cur_state_lst):
            cur_entity = state_dict['entity']

            if cur_entity.lower() not in visited_entities:
                visited_entities.append(cur_entity.lower())
            else:
                delete_idx.append(tuple((cur_entity.lower(), idx)))

        if delete_idx:
            ind = 0
            for delete_tup in delete_idx:
                cur_entity, cur_idx = delete_tup

                if 'wii' in cur_entity:
                    ind = 1

                temp_answer = cur_state_lst[cur_idx]['answers']
                for idx, entity_dict in enumerate(cur_state_lst):
                    if entity_dict['entity'].lower() == cur_entity.lower():
                        for step_id, step_content in temp_answer.items():
                            cur_state_lst[idx]['answers'][step_id] += step_content
                        break

            delete_idx = [item[1] for item in delete_idx]
            original_data[key]['states'] = [item for (idx, item) in enumerate(cur_state_lst) if idx not in delete_idx]
            
    save_json(original_data, args.out_path) 

if __name__ == '__main__':
    main()
    

