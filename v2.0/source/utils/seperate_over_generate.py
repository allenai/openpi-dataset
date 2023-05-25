import json
from os import walk


def main():
    entity_data = json.load(open('../../data/data_in_new_format/dev-data-reformatted-v4-entity-overgenerated.json'))
    attr_data = json.load(open('../../data/data_in_new_format/dev-data-reformatted-v4-attribute-overgenerated.json'))
    original_data = json.load(open('../../data/data_in_new_format/dev-data-reformatted-v4.json'))

    for proc_id, proc_content in original_data.items():
        cur_cluster = proc_content['clusters']
        entity_cluster = entity_data[proc_id]['clusters']
        attr_cluster = attr_data[proc_id]['clusters']

        for entity_id, entity_content in cur_cluster.items():
            # seperate originally anntoated entity with newly generated entity
            cur_original_entity_cluster = entity_content['entity_cluster']
            cur_entity_cluster = entity_cluster[entity_id]['entity_cluster']

            new_entity = [item for item in cur_entity_cluster if item not in cur_original_entity_cluster]

            original_data[proc_id]['clusters'][entity_id]['entity_overgenerate'] = new_entity 
            original_data[proc_id]['clusters'][entity_id]['entity_cluster_combined'] = cur_entity_cluster 

            cur_original_attr_dict = entity_content['attribute_cluster']
            for attr_id, attr_content in cur_original_attr_dict.items():
                cur_attr_cluster = attr_cluster[entity_id]['attribute_cluster'][attr_id]

                new_attr = [item for item in cur_attr_cluster if item not in attr_content]

                original_data[proc_id]['clusters'][entity_id]['attribute_cluster'][attr_id] = {
                    'attribute_cluster': attr_content,
                    'attribute_overgenerate': new_attr,
                    'attribute_cluster_combined': cur_attr_cluster
                } 

    json.dump(
        original_data, 
        open('../../data/data_in_new_format/dev-data-reformatted-v4-overgenerated.json', 'w'), 
        indent=4
    )


if __name__ == '__main__':
    main()
