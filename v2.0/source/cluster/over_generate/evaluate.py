import json


def main():
    ent_data = json.load(open('../../../data/over_generate_relevance_annotation/v4-entity-annotation.json', 'r'))
    attr_data = json.load(open('../../../data/over_generate_relevance_annotation/v4-attribute-annotation.json', 'r'))

    ent_count, og_ent_count, og_ent_wrong, ent_complete_correct = 0, 0, 0, 0
    
    for ent_val in ent_data.values():
        cur_annotation = ent_val['annotation']

        for cur_ent in cur_annotation.values():
            if cur_ent['overgenerate_cluster'] is not None:
                og_ent_count += len(cur_ent['overgenerate_cluster'])
                og_ent_wrong += len(cur_ent['bad_overgenerate'])
                ent_count += 1
                if cur_ent['bad_overgenerate'] == []:
                    ent_complete_correct += 1

    attr_count, og_attr_count, og_attr_wrong, attr_complete_correct = 0, 0, 0, 0

    for attr_val in attr_data.values():
        cur_annotation = attr_val['annotation']

        for cur_attr in cur_annotation.values():
            attr_annotation = cur_attr['attribute_annotation']
            for annot_dict in attr_annotation:
                if annot_dict['attribute_overgenerate'] is not None:
                    og_attr_count += len(annot_dict['attribute_overgenerate'])
                    og_attr_wrong += len(annot_dict['bad_overgenerate'])
                    attr_count += 1
                    if annot_dict['bad_overgenerate'] == []:
                        attr_complete_correct += 1

    print('-----------------------------------------------------------')
    print(f'Entity Overgenerate Accuracy: {(1 - og_ent_wrong / og_ent_count):.3f}')
    print(f'Entity Overgenerate Completely Correct Proportion: {(ent_complete_correct / ent_count):.3f}')
    print('-----------------------------------------------------------')
    print(f'Attribute Overgenerate Accuracy: {(1 - og_attr_wrong / og_attr_count):.3f}')
    print(f'Attribute Overgenerate Completely Correct Proportion: {(attr_complete_correct / attr_count):.3f}')
    print('-----------------------------------------------------------')


if __name__ == "__main__":
    main()
