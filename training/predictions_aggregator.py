# input = filepath (of the file dumped by predictor)
# id, ques, answer1
# id, ques, answer2

# output = filepath of aggregated outputs
# id, ques, [answer1, answer2]
import argparse
import itertools
import json
from typing import Dict, List

from eval.eval_util import sort_map_by_key

DEFAULT_ANS = "There will be no change."


def read_jsonl(input_file: str, strip_lines=False) -> List[Dict]:
    output: List[Dict] = []
    with open(input_file, 'r') as open_file:
        for line in open_file:
            output.append(json.loads(line if not strip_lines else line.strip()))
    return output

def aggregate_predictions(prediction_fp: str, out_fp: str, separator="||", window_size: int=None, state_change_sep:str=".. And "):
    if window_size and window_size > 1:
        return aggregate_windows(prediction_fp=prediction_fp, out_fp=out_fp)

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
        if a.count(scsep) > 1:
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


def aggregate_windows(prediction_fp: str, out_fp: str, separator:str="||", state_change_sep:str=".. And "):
    input_jsonl: List[Dict[str, List[str]]] = read_jsonl(prediction_fp)
    # example of an id is: "wikihow.com/...||1||1||2"
    #   indicating wikihow article's sentence 1, 2 and 3 together.
    # Aggregate by url (e.g., "wikihow.com/...) to get an array of state changes

    # {"www.wikihow.com/Repurpose-Household-Items-with-Paint||1||2": [{"answer": "screen was dull before, and bright afterwards.. And light was dim before, and bright afterwards.. An. ", "id": "www.wikihow.com/Repurpose-Household-Items-with-Paint||1"}, {"answer": "car was dull before, and bright afterwards.. And car was cold before, and warm afterwards.. An. ", "id": "www.wikihow.com/Repurpose-Household-Items-with-Paint||2"}]}

    # lambda xx: "wikihow.com/xyz||1||1||2" => "wikihow.com/xyz"
    # Due to awkwardly shaped json: it is not hard to extract the url
    xx = lambda one_jsonl: one_jsonl["window_key"].split(separator)[0]
    fmt = {}

    with open(out_fp, 'w') as outfile:
        for item in input_jsonl:
            window_idx = idx_to_window_idx(item['window_key'])
            window_answers = [x['answer'] for x in item['window_values']]
            window_idx_ans = zip(window_idx, window_answers)
            for sent_id, id_ans_lst in list(window_idx_ans):
                # id_ans_list looks like this.
                # [
                # {'id': 'www.wikihow.com/Repurpose-Household-Items-with-Paint||1',
                #  'answer': 'and then, A was dull before, and bright afterwards'
                #            '.. And B was red before, and white afterwards'
                #  },
                # {'id': 'www.wikihow.com/Repurpose-Household-Items-with-Paint||1',
                #  'answer': '...'
                #  }
                # ]
                for sc in ans_to_list(id_ans_lst, state_change_seps=[state_change_sep, ","]):
                    if sent_id not in fmt:
                        fmt[sent_id] = {"id": sent_id, "answers": []}

                    fmt[sent_id]["answers"].append(sc)

            # append lines to jsonl.
        for k, v in sort_map_by_key(fmt, reverse_order=False).items():
            v['answers'] = list(set(v['answers']))
            print(json.dumps(v), file=outfile)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="Path to the unaggregated file e.g. data/trackworld/tw_bench/tw_bench_propara_npn_ea.jsonl."
    )
    parser.add_argument(
        "--output_path", "-o", required=True, type=str,
        help="Path to the aggregated file. e.g. data/trackworld/tw_bench/tw_bench_propara_npn_ea_aggregated.jsonl"
    )
    parser.add_argument(
        "--window_size", "-w", required=False, default=None, type=int,
        help="Some outputs contain ids like so: wikihow.com/.../1||2||3 with "
             "state changes for k sentences (here, 1,2,3)"
    )

    args = parser.parse_args()

    if not args.window_size or args.window_size <= 1:
        aggregate_predictions(prediction_fp=args.input_path, out_fp=args.output_path)
    else:
        aggregate_windows(prediction_fp=args.input_path, out_fp=args.output_path)
    print(f"Output is in {args.output_path}")


if __name__ == "__main__":
    main()
