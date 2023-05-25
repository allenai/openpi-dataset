# input = filepath (of the file dumped by predictor)
# id, ques, answer1
# id, ques, answer2

# output = filepath of aggregated outputs
# id, ques, [answer1, answer2]
import argparse
import itertools
import json
from typing import Dict, List

DEFAULT_ANS = "There will be no change."


def read_jsonl(input_file: str, strip_lines=False) -> List[Dict]:
    output: List[Dict] = []
    with open(input_file, 'r') as open_file:
        for line in open_file:
            output.append(json.loads(line if not strip_lines else line.strip()))
    return output


def aggregate_predictions(prediction_fp: str, out_fp: str, separator="||", state_change_sep: str = ".. And "):
    json_lines = read_jsonl(prediction_fp)
    # example of an id is: "wikihow.com/...||1||1||2"
    #   might indicate wikihow article's sentence 1, entity 1, attribute 2
    # Aggregate by sentence (e.g., "wikihow.com/...||1) to get an array of state changes
    with open(out_fp, 'w') as outfile:
        for grp_id, grp in itertools.groupby(json_lines, lambda line: separator.join(line["id"].split(separator)[0:2])):
            fmt_j = {"id": grp_id, "answers": []}
            for line in grp:
                # This is in the format required by the evaluator.
                for a in ans_to_list(line['answer'], state_change_seps=[state_change_sep, ","]):
                    if not a.lower().strip().startswith("there will be no change"):
                        fmt_j["answers"].append(a)
            if not fmt_j["answers"]:
                fmt_j["answers"].append(DEFAULT_ANS)
            print(json.dumps(fmt_j), file=outfile)


def filter_answers(answers):
    answers = [x.strip() for x in answers if accept_answer(x)]
    if not answers:
        answers = [DEFAULT_ANS]
    return answers


def accept_answer(answer, keywords=['of', 'was', 'afterwards', 'before']):
    for kw in keywords:
        if kw not in answer:
            return False
    return True


def ans_to_list(a, state_change_seps):
    answers = []
    for scsep in state_change_seps:
        if a.count(scsep) > 0:
            answers = a.split(scsep)
            return filter_answers(answers)
    # if no separator was found or one sep was found.
    answers = [a]
    return filter_answers(answers)


def idx_to_window_idx(idx: str) -> List[str]:
    split_idx = idx.split('||')
    passage_idx = split_idx[0]

    output_window_idx = []
    for window_item in split_idx[1:]:
        win_idx = passage_idx + '||' + window_item
        output_window_idx.append(win_idx)
    return output_window_idx


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="Path to the unformatted file e.g. data/predictions.jsonl."
    )
    parser.add_argument(
        "--output_path", "-o", required=True, type=str,
        help="Path to the formatted file. e.g. data/predictions_as_list.jsonl"
    )

    args = parser.parse_args()

    aggregate_predictions(prediction_fp=args.input_path, out_fp=args.output_path)
    print(f"Output is in {args.output_path}")


if __name__ == "__main__":
    main()
