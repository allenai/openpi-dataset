import json


def main():
    data = json.load(open('../../data/data_in_new_format/dev-data-reformatted-v4-overgenerated.json', 'r'))

    for proc_id, proc_content in data.items():
        cur_clusters = proc_content['clusters']

        for ent_id, ent_content in cur_clusters.items():
            attr_clusters = ent_content['attribute_cluster']

            for attr_id, attr_cluster in attr_clusters.items():

                data[proc_id]['clusters'][ent_id]['attribute_cluster'][attr_id]['attribute_overgenerate'] = list(set(attr_cluster['attribute_overgenerate']))
                data[proc_id]['clusters'][ent_id]['attribute_cluster'][attr_id]['attribute_cluster_combined'] = list(set(attr_cluster['attribute_cluster_combined']))

    data = json.dump(
        data,
        open('../../data/data_in_new_format/dev-data-reformatted-v4-overgenerated-unique.json', 'w'), 
        indent=4
    )

if __name__ == '__main__':
    main()
