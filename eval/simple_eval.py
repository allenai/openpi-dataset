import json
import os
import sys
from typing import List, Callable, Tuple
import datetime

from tqdm import tqdm

from eval.eval_util import normalize_nostem
from eval.generation_metric import GenerationMetric, ExactMetric, BLEUMetric, ROUGEMetric
from eval.eval_dataset_reader import PredictionsFileReader

EFFECT_STOP_WORDS = {"and", "was", "is", "before", "afterwards", "after", "of"}


def get_content_from_predicted_effect(s):
    def is_stop(cand: str):
        return cand.lower() in EFFECT_STOP_WORDS

    # if s and skip_attr_in_ans:  # if non-zero len.
    #     # before: "age of person was xx before and yy after."
    #     # after this code block: "person was xx before and yy after."
    #     attr_signature = ' of '
    #     if attr_signature in s:
    #         s = s[s.find(attr_signature) + len(attr_signature):]
    words_prev = normalize_nostem(s).split(" ")
    words = []
    for word in words_prev:
        if not is_stop(word):
            words.append(word)
    return ' '.join(words)


def f1_emnlp2020(
        predictions: List[str],
        gold: List[str],
        generation_metric: GenerationMetric) -> Tuple[float, float, float]:
    if len(gold) == 0 and len(predictions) == 0:
        return (1.0, 1.0, 1.0)
    if len(gold) == 0 and len(predictions) > 0:
        return (0.0, 1.0, 0.0)
    if len(predictions) == 0:
        return (1.0, 0.0, 0.0)

    predictions = list(set(predictions))
    gold = list(set(gold))
    
    # Compute precision score based on best gold match for each prediction
    tp = 0.0  # true positive score
    for p in predictions:
        best_gold_match = 0.0
        norm_p = get_content_from_predicted_effect(p)
        for g in gold:
            norm_g = get_content_from_predicted_effect(g)
            gold_match = generation_metric.match_score(gold=norm_g, predicted=norm_p)
            if gold_match > best_gold_match:
                best_gold_match = gold_match
        # print(f"best_gold_match :{best_gold_match}")
        tp += best_gold_match
    precision = tp / len(predictions)

    # Compute recall score based on best prediction match for each gold
    tr = 0.0
    for g in gold:
        norm_g = get_content_from_predicted_effect(g)
        best_prediction_match = 0.0
        for p in predictions:
            norm_p = get_content_from_predicted_effect(p)
            prediction_match = generation_metric.match_score(gold=norm_g, predicted=norm_p)
            if prediction_match > best_prediction_match:
                best_prediction_match = prediction_match
        tr += best_prediction_match
    recall = tr / len(gold)

    f1_denominator = precision + recall

    if f1_denominator == 0:
        return (0.0, 0.0, 0.0)
    return (precision, recall, 2 * precision * recall / (precision + recall))


class SizeMismatch(Exception):
    pass


def evaluate(predictions_reader: PredictionsFileReader,
             gold_answers_reader: PredictionsFileReader,
             diag: Callable[[str], None],
             generation_metric: GenerationMetric) -> dict:
    if len(predictions_reader.get_all_question_ids()) != len(gold_answers_reader.get_all_question_ids()):
        raise SizeMismatch(
            f"Error: Size mismatch: {predictions_reader.in_path} has {len(predictions_reader.get_all_question_ids())} predictions and \n{gold_answers_reader.in_path} has {len(gold_answers_reader.get_all_question_ids())} answers."
        )

    metric_main_p_sum = 0.0
    metric_main_r_sum = 0.0
    metric_main_f1_sum = 0.0

    all_q_ids = gold_answers_reader.get_all_question_ids()
    for q_id in tqdm(all_q_ids):
        predictions = predictions_reader.get_answers_for_id(id=q_id)
        if len(predictions) == 1 and predictions[0].lower().strip().startswith("there will be no change"):
            predictions = []
        gold_answers = gold_answers_reader.get_answers_for_id(id=q_id)

        diag("Prediction:")
        diag(predictions)
        diag("Gold:")
        diag(gold_answers)
        diag("")

        # Main metric
        (p, r, f1) = f1_emnlp2020(
            predictions=predictions,
            gold=gold_answers,
            generation_metric=generation_metric
        )
        metric_main_p_sum += p
        metric_main_r_sum += r
        metric_main_f1_sum += f1
        diag("Instance Metrics:")
        diag(f"    Main:  P={p}, R={r}, F1={f1}")
        diag("")

    return {
        "main_P": '%.2f' % (100.0 * metric_main_p_sum / len(all_q_ids)),
        "main_R": '%.2f' % (100.0 * metric_main_r_sum / len(all_q_ids)),
        "main_F1": '%.2f' % (100.0 * metric_main_f1_sum / len(all_q_ids)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate OpenPI predictions.')

    parser.add_argument(
        '--gold-file', '-g',
        help='Filename with gold answers',
        required=True)

    parser.add_argument(
        '--prediction-file', '-p',
        help='Filename with predictions',
        required=True)

    parser.add_argument(
        '--diagnostics_file', '-d',
        help='If provided, diagnostic will be printed in a file (default: %(default)s)',
        default=f"/tmp/diagnostics_{str(datetime.datetime.now()).split('.')[0].replace('-', '_').replace(':', '_').replace(' ', '__')}.txt",
        required=False)

    parser.add_argument(
        '--quiet', '-q',
        help='If provided, diagnostic will not be printed or written to file.',
        action='store_true',
        required=False)

    parser.add_argument(
        '--output', '-o',
        help='Output metrics to this file in JSON format. If not specified, metrics are only printed to stdout as JSON.',
        default=None,
        required=False)

    args = parser.parse_args()
    diagnostics_file = open(args.diagnostics_file, 'w')

    def diag(msg: str):
        if args.quiet:
            return
        diagnostics_file.write(f"{msg}")
        diagnostics_file.write("\n")

    if not args.gold_file or not os.path.exists(args.gold_file):
        print(f"WARNING: Not performing any evaluation because input gold file does not exist: {args.gold_file}")
        return

    if not args.prediction_file or not os.path.exists(args.prediction_file):
        print(f"WARNING: Not performing any evaluation because prediction file does not exist: {args.prediction_file}")
        return

    predictions = PredictionsFileReader(in_path=args.prediction_file)
    gold_answers = PredictionsFileReader(in_path=args.gold_file)

    generation_metrics = [
        ExactMetric(),
        BLEUMetric(),
        ROUGEMetric()
    ]
    output = open(args.output, "w", encoding="UTF-8") if args.output else sys.stdout

    all_metrics = dict()
    formatted_scores = []

    for metric_num, current_metric in enumerate(generation_metrics):
        print(f"\nEvaluating current metric ({1 + metric_num}/{len(generation_metrics)}) : {current_metric.name()} ...")
        current_metric_score = evaluate(predictions_reader=predictions,
                                        gold_answers_reader=gold_answers,
                                        diag=diag,
                                        generation_metric=current_metric
                                        )
        for k, v in current_metric_score.items():
            # prepare all metrics as json entries.
            all_metrics[f"{k.replace('main_', '')}_{current_metric.name()}"] = v
        formatted_scores.append(f"{current_metric.name()}"
                                f"\t{current_metric_score['main_P']}"
                                f"\t{current_metric_score['main_R']}"
                                f"\t{current_metric_score['main_F1']}"
                                )

    if args.output:
        json.dump(all_metrics, output)

    print(f"\n\n================================\n Evaluation results \n================================")
    print(f"Predictions: {args.prediction_file}")
    print(f"Gold: {args.gold_file}")
    if not args.quiet:
        print(f"Diagnostics: {args.diagnostics_file}")
    print(f"\n\t\tprec\trecall\tf1")
    for fs in formatted_scores:
        print(fs)

    output.close()
    diagnostics_file.close()


if __name__ == '__main__':
    main()
