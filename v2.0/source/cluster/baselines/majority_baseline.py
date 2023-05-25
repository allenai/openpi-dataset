import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_path', type=str, required=True
    )
    return parser.parse_args()


def main():
    args = get_args()
    gt_labels = pickle.load(open(args.gt_path, 'rb'))

    print(sum(gt_labels))
    print(len(gt_labels))
    if sum(gt_labels) >= (len(gt_labels)//2):
        pred_labels = [1] * len(gt_labels)
    else:
        pred_labels = [0] * len(gt_labels)

    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average='macro')

    print(f'Accuracy score: {acc}')
    print(f'F1 score: {f1}')


if __name__ == "__main__":
    main()
