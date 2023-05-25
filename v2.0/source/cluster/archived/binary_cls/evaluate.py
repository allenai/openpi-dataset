import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predict_path', type=str, required=True
    )
    parser.add_argument(
        '--gt_path', type=str, required=True
    )
    return parser.parse_args()


def main():
    args = get_args()
    pred_results = pickle.load(open(args.predict_path, 'rb'))
    gt_results = pickle.load(open(args.gt_path, 'rb'))

    assert len(pred_results) == len(gt_results)

    if isinstance(pred_results[0], str):
        if 'chat' in args.predict_path:
            pred_results = [1 if ele.lower() == 'yes' else 0 for ele in pred_results]
        else:
            pred_results = [1 if ele.lower() == 'true' else 0 for ele in pred_results]

    acc = accuracy_score(gt_results, pred_results)
    f1 = f1_score(gt_results, pred_results, average='macro')
    conf_mtx = confusion_matrix(gt_results, pred_results, labels=[0, 1])

    print(f'Accuracy score: {acc}')
    print(f'F1 score: {f1}')

    conf_mtx_plot = ConfusionMatrixDisplay(conf_mtx, display_labels = [0, 1])
    conf_plot = conf_mtx_plot.plot()
    fname = args.predict_path.split('/')[-1].split('.')[0]
    plt.savefig(f'./plots/{fname}-conf-mtx.png')


if __name__ == '__main__':
    main()
