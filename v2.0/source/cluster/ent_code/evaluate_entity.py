import json
import argparse
import numpy as np
from glob import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_dir', type=str, required=True, help='path to the result file'
    )
    return parser.parse_args()


def main():
    args = get_args()
    result_dir = glob(args.result_dir + '/*.json')
    result_dir = [path for path in result_dir if 'run' in path]
    # gt_data = json.load(open('../data/dev_data_mini_annotated_entity.json'))
    gt_data = json.load(open('../archived/comparison_data/dev_data_mini_annotated_entity.json'))

    run_3_p, run_3_r, run_3_f = [], [], []
    run_5_p, run_5_r, run_5_f = [], [], []

    for result_path in result_dir:
        result = json.load(open(result_path, 'r'))
        pred_in_gt, gt_in_pred, total_gt, total_pred = 0, 0, 0, 0
        for proc_id, pred_cluster in result.items():

            gt_cluster = gt_data[proc_id]['entities']
            gt_cluster = [[item.strip() for item in sublst] for sublst in gt_cluster]
            gt_entities = [item for sublst in gt_cluster for item in sublst]
            pred_cluster = list(pred_cluster.values())
            pred_cluster = [[item.strip() for item in sublst] for sublst in pred_cluster]
            pred_cluster = [[item for item in sublst if item in gt_entities] for sublst in pred_cluster]

            total_gt += len(gt_cluster)
            total_pred += len(pred_cluster)

            for pred in pred_cluster:
                for gt_item in gt_cluster:
                    if (sum([item in gt_item for item in pred]) == len(gt_item)) and (len(pred) == len(gt_item)):
                        pred_in_gt += 1
                        break

            for gt in gt_cluster:
                for pred_item in pred_cluster:
                    if (sum([item in pred_item for item in gt]) == len(pred_item)) and (len(gt) == len(pred_item)):
                        gt_in_pred += 1
                        break
        
        recall = pred_in_gt / total_gt
        precision = gt_in_pred / total_pred
        f1 = 2 * recall * precision / (recall + precision)

        if result_path.split('_')[-2] == '5':
            run_5_r.append(recall)
            run_5_p.append(precision)
            run_5_f.append(f1)
        else:
            run_3_r.append(recall)
            run_3_p.append(precision)
            run_3_f.append(f1)

    print(f'Average Recall for num_run=3: {np.mean(run_3_r):.3f}')
    print(f'Average Precision for num_run=3: {np.mean(run_3_p):.3f}')
    print(f'Average F1 for num_run=3: {np.mean(run_3_f):.3f}')

    print('----------------------------------')

    print(f'Average Recall for num_run=5: {np.mean(run_5_r):.3f}')
    print(f'Average Precision for num_run=5: {np.mean(run_5_p):.3f}')
    print(f'Average F1 for num_run=5: {np.mean(run_5_f):.3f}')

if __name__ == '__main__':
    main()

