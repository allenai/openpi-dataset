# for (dev, test, train) merge the following:
# id, answers_metadata              data/gold/dev/id_answers_metadata.jsonl
# id, answers                       data/gold/dev/id_answers.jsonl
# id, question_metadata             data/gold/dev/id_question_metadata.jsonl
# id, question                      data/gold/dev/id_question.jsonl
import json
from typing import List, Dict, Any

from visual.bug import patch_issue_8


def load(in_filepath: str) -> List[Dict[str, Any]]:
    d: List[Dict[str, Any]] = []
    with open(in_filepath, 'r') as infile:
        for line in infile:
            if line:
                d.append(json.loads(line.strip()))
    return d


if __name__ == '__main__':
    patches = patch_issue_8()  # 263 (out of 29K) examples had empty answers. Fix them.

    for p in ["train", "dev", "test"]:
        os = []
        for q, a, qm, am in zip(load(f"data/gold/{p}/id_question.jsonl"), load(f"data/gold/{p}/id_answers.jsonl"), load(f"data/gold/{p}/id_question_metadata.jsonl"), load(f"data/gold/{p}/id_answers_metadata.jsonl")):
            assert q['id'] == a['id'] and a['id'] == qm['id'] and qm['id'] == am['id'], f"ID issue for {q}"
            o = {}
            o['id'] = q['id']
            o['question'] = q['question']
            bug_to_fix = o['id'] in patches
            if bug_to_fix:
                patch = patches[o['id']]
                o['answers'] = patch['answers']
                if not qm['question_metadata'] or "url" not in qm['question_metadata'] or not qm['question_metadata']["url"]:
                    o['question_metadata'] = patch['question_metadata']
                else:
                    o['question_metadata'] = qm['question_metadata']
                o['answers_metadata'] = patch['answers_metadata']
            else:
                o['answers'] = a['answers']
                o['question_metadata'] = qm['question_metadata']
                o['answers_metadata'] = am['answers_metadata']
            os.append(o)
        with open(f"data/gold_combined/{p}.jsonl", 'w') as outfile:
            for o in os:
                outfile.write(json.dumps(o))
                outfile.write("\n")

        print(f"\n output in data/gold_combined/")