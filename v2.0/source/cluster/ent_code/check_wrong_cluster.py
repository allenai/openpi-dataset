import os 
import ast
import json
import argparse
from glob import glob
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_dir', type=str, required=True, help='path to the result file'
    )
    parser.add_argument(
        '--num_shot', type=str, default='3', help='number of shot'
    )
    return parser.parse_args()


def main():
    args = get_args()
    result_dir = glob(args.result_dir + '/*.json')
    result_dir = [item for item in result_dir if item.split('_')[-2] == args.num_shot and 'error' not in item]
    gt_data = json.load(open('../data/dev_data_mini_annotated_entity.json'))

    all_mistakes = []
    for result_path in result_dir:
        result = json.load(open(result_path, 'r'))

        counter = 0
        error_dict = {}
        for proc_id, pred_cluster in result.items():

            gt_cluster = gt_data[proc_id]['entities']
            gt_cluster = [[item.strip() for item in sublst] for sublst in gt_cluster]
            gt_entities = [item for sublst in gt_cluster for item in sublst]
            pred_cluster = list(pred_cluster.values())
            pred_cluster = [[item.strip() for item in sublst] for sublst in pred_cluster]
            pred_cluster = [[item for item in sublst if item in gt_entities] for sublst in pred_cluster]

            for pred_c in pred_cluster:
                for gt_c in gt_cluster:
                    cur_correct = sum([item in gt_c for item in pred_c])
                    if cur_correct > 0 and cur_correct != len(gt_c):
                        all_mistakes.append(tuple((str(gt_c), str(pred_c))))

    error_dict = {}
    counter = 0
    for item, count in Counter(all_mistakes).most_common(20):
        cur_gt = ast.literal_eval(item[0])
        cur_pred = ast.literal_eval(item[1])
        error_dict[str(counter)] = {
            'gt': cur_gt,
            'pred': cur_pred,
            'count': count
        }
        counter += 1

    f_name = f'{args.num_shot}_common_error.json'
    with open(os.path.join(args.result_dir, f_name), 'w') as f:
        json.dump(error_dict, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()

