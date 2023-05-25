import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path', type=str, required=True, help='path to the result file'
    )
    return parser.parse_args()


def main():
    args = get_args()
    result_dict = json.load(open(args.result_path, 'r'))
    # gt_dict = json.load(open('../data/dev_data_mini_annotated_attr.json', 'r'))
    gt_dict = json.load(open('../archived/comparison_data/dev_data_mini_annotated_attr.json', 'r'))

    gt_in_pred, pred_in_gt, total_gt, total_pred = 0, 0, 0, 0
    for proc_id, result in result_dict.items():
        gt = gt_dict[proc_id]['entity_attr_dict']
        for entity, cluster in result.items():
            gt_cluster = [item.split(' | ') for item in gt[entity]]
            gt_cluster = [[item.strip() for item in sublst] for sublst in gt_cluster]

            gt_attr = [lst for sublst in gt_cluster for lst in sublst]

            cur_cluster = []
            for key, val in cluster.items():
                val = [item.strip() for item in val]
                val = [item for item in val if item in gt_attr]
                if key in gt_attr:
                    cur_cluster.append([key] + val)
                else:
                    cur_cluster.append(val)

            if len(gt_cluster) == 1:
                continue

            cur_cluster = [list(set(item)) for item in cur_cluster]
            total_pred += len(cur_cluster)
            total_gt += len(gt_cluster)

            for pred_c in cur_cluster:
                for gt_c in gt_cluster:
                    if (sum([item in gt_c for item in pred_c]) == len(gt_c)) and (len(pred_c) == len(gt_c)):
                        pred_in_gt += 1
                        break

            for gt_c in gt_cluster:
                for pred_c in cur_cluster:
                    if (sum([item in pred_c for item in gt_c]) == len(pred_c)) and (len(pred_c) == len(gt_c)):
                        gt_in_pred += 1
                        break

    recall = pred_in_gt / total_gt
    precision = gt_in_pred / total_pred
    f1 = 2 * precision * recall / (precision + recall)

    print(f'recall: {recall:.3f}')
    print(f'precision: {precision:.3f}')
    print(f'f1: {f1:.3f}')


if __name__ == '__main__':
    main()
