import json
from glob import glob


def main():
    file_paths = glob('../../llama-65/*.json')

    pred_paths = [path for path in file_paths if 'pred' in path]
    gold_paths = [path for path in file_paths if 'gold' in path]

    pred_data = {}
    gold_data = {}

    for pred_path in pred_paths:
        cur_data = json.load(open(pred_path, 'r'))
        for id, content in cur_data.items():
            pred_data[id] = content

    for gold_path in gold_paths:
        cur_data = json.load(open(gold_path, 'r'))
        for id, content in cur_data.items():
            gold_data[id] = content

    with open('../../data/dev_states_llama-65b.json', 'w') as f:
        json.dump(pred_data, f, indent=4)
    f.close()

    with open('../../data/dev_states_llama-65b_gold.json', 'w') as f:
        json.dump(gold_data, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()