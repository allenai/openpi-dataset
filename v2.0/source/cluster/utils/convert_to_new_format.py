import json


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def remove_determinant(word: str) -> str:
    if word.startswith('the '):
        return word[4:]
    elif word.startswith('a '):
        return word[2:]
    elif word.startswith('an '):
        return word[3:]
    else:
        return word


def plural_to_singular(word: str) -> str:
    if not word.endswith('ss'):
        if word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('s'):
            return word[:-1]
        elif word.endswith('es'):
            return word[:-2]
        else:
            return word
    else:
        return word


def main():
    original_data = load_json('../../../../data/dev-ranked.json')
    entity_dict = load_json('../ent_code/gpt_cluster/results/dev_entity_chat_result_5.json')
    attr_dict = load_json('../attr_code/gpt_cluster/results/dev_attr_chat_result_3.json')

    for proc_id, meta_content in original_data.items():
        cur_state_lst = meta_content['states']
        proc_entity_dict = entity_dict[proc_id]
        proc_attr_dict = attr_dict[proc_id]

        # a round of cleaning of the GPT attribute cluster dictionary
        for entity_id, entity_state_dict in proc_attr_dict.items():

            new_entity_state_dict = {}
            for attr_id, attr_content in entity_state_dict.items():
                attr_content = [item.split(' | ') for item in attr_content]
                attr_content = [item for sublist in attr_content for item in sublist]
                attr_content = [item.strip() for item in attr_content]
                new_entity_state_dict[attr_id.split(' | ')[0]] = list(set(attr_content + [attr_id.split(' | ')[0]]))

            proc_attr_dict[entity_id] = new_entity_state_dict

        # map annotated attribute name to their cannonical form
        for state_id, entity_state in enumerate(cur_state_lst):
            cur_entity = entity_state['entity']
            cur_answers = entity_state['answers']
            cur_attr_dict = proc_attr_dict[cur_entity]
            cur_attr_dict = {k: [k] + v for k, v in cur_attr_dict.items()}

            for step_id, step_state_lst in cur_answers.items():
                for list_idx, step_state in enumerate(step_state_lst):
                    cur_attr = step_state['attribute']
                    cur_attr_lst = [item.strip() for item in cur_attr.split(' | ')]

                    ind = 0
                    for attr in cur_attr_lst:
                        for attr_id, attr_cluster in cur_attr_dict.items():
                            if attr in attr_cluster:
                                original_data[proc_id]['states'][state_id]['answers'][step_id][list_idx]['attribute_annotation'] = cur_attr
                                original_data[proc_id]['states'][state_id]['answers'][step_id][list_idx]['attribute'] = attr_id
                                ind = 1
                                break
                        if ind:
                            break

                    if not ind:
                        raise ValueError('Attribute not found in the attribute cluster dictionary.')

        # map annotated entity name to their cannonical form
        for state_id, entity_state in enumerate(cur_state_lst):
            cur_entity = entity_state['entity']
            cur_entity_lst = [item.strip() for item in cur_entity.split(' | ')]

            ind = 0
            for entity in cur_entity_lst:
                for entity_id, entity_cluster in proc_entity_dict.items():
                    if entity in entity_cluster:
                        original_data[proc_id]['states'][state_id]['entity_annotation'] = [cur_entity]
                        original_data[proc_id]['states'][state_id]['entity'] = entity_id
                        ind = 1
                        break 
                if ind:
                    break 

            if not ind:
                for entity in cur_entity_lst:
                    nd_entity = remove_determinant(entity)
                    for entity_id, entity_cluster in proc_entity_dict.items():
                        entity_cluster += [remove_determinant(item) for item in entity_cluster] + [plural_to_singular(item) for item in entity_cluster]
                        if nd_entity in entity_cluster:
                            original_data[proc_id]['states'][state_id]['entity_annotation'] = [cur_entity]
                            original_data[proc_id]['states'][state_id]['entity'] = entity_id
                            proc_entity_dict[entity_id] = list(set(entity_cluster))
                            ind = 1

            if not ind:
                raise ValueError('Entity not found in the entity cluster dictionary.')

        # add attribute to entity cluster
        proc_entity_dict = entity_dict[proc_id]
        proc_attr_dict = attr_dict[proc_id]

        new_cluster_dict = {}
        for entity_id, entity_cluster in proc_entity_dict.items():

            temp_dict = {
                'entity_cluster': entity_cluster,
                'attribute_cluster': {}
            }
            for entity_name, attr_cluster in proc_attr_dict.items():
                cur_entity_name_lst = [item.strip() for item in entity_name.split(' | ')]
                for e in cur_entity_name_lst:
                    if e == entity_id or e in entity_cluster:
                        for k, v in attr_cluster.items():
                            temp_dict['attribute_cluster'][k] = v

            new_cluster_dict[entity_id] = temp_dict

        original_data[proc_id]['clusters'] = new_cluster_dict

        # add attribute to new clusters
        for state_id, state_content in enumerate(cur_state_lst):
            cur_entity = state_content['entity']
            # current cluster of the entity
            cur_entity_cluster = new_cluster_dict[cur_entity]['entity_cluster']

            candidate_dict = {}
            counter = 0
            for entity in cur_entity_cluster:
                cur_state_lst = original_data[proc_id]['states']
                for state_dict in cur_state_lst:
                    try:
                        if entity in new_cluster_dict[state_dict['entity']]['entity_cluster']:
                            candidate_dict[str(counter)] = state_dict['answers']
                            counter += 1
                    except:
                        raise ValueError('Entity not found in the entity cluster dictionary.')

            total_steps = len(candidate_dict['0'])
            new_state_dict = {f'step{i+1}': [] for i in range(total_steps)}
            visited_attr = {f'step{i+1}': [] for i in range(total_steps)}

            for entity, ans_dict in candidate_dict.items():
                for i in range(total_steps):
                    cur_step = f'step{i+1}'
                    cur_ans = ans_dict[cur_step]
                    for content in cur_ans:
                        if content['attribute'] not in visited_attr[cur_step]:
                            new_state_dict[cur_step].append(content)
                            visited_attr[cur_step].append(content['attribute'])

            original_data[proc_id]['states'][state_id]['answers'] = new_state_dict

    # remove dupicated entities in answers
    for proc_id, proc_content in original_data.items():
        cur_state_dict = proc_content['states']
        visited_entity, remove_idx = [], []
        for cur_idx, entity_state_dict in enumerate(cur_state_dict):
            cur_entity = entity_state_dict['entity']
            cur_entity_annotation = entity_state_dict['entity_annotation']

            if cur_entity not in visited_entity:
                visited_entity.append(cur_entity)
            else:
                sub_state_dict = cur_state_dict[:cur_idx]
                ind = 0
                for sub_idx, sub_entity_state_dict in enumerate(sub_state_dict):
                    if sub_entity_state_dict['entity'] == cur_entity:
                        original_data[proc_id]['states'][sub_idx]['entity_annotation'].extend(cur_entity_annotation)
                        ind = 1
                        break
                if not ind:
                    raise ValueError('Entity not found in the previous states during "entity_annotation" clustering.')

                remove_idx.append(cur_idx)

        new_state_dict = [cur_state_dict[i] for i in range(len(cur_state_dict)) if i not in remove_idx]
        original_data[proc_id]['states'] = new_state_dict

    # another round of cleaning for the cluster
    for proc_id, proc_content in original_data.items():
        cur_states = proc_content['states']
        cur_clusters = proc_content['clusters']

        # remove duplicated entities
        for cluster_key, cluster_content in cur_clusters.items():
            original_data[proc_id]['clusters'][cluster_key]['entity_cluster'] = list(set(cluster_content['entity_cluster']))
    
    with open('../../../../data/data_in_new_format/dev-data-reformatted-v4.json', 'w') as f:
        json.dump(original_data, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()
