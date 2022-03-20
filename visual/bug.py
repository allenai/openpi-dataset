import json
from typing import Dict, Any

from training.gen_ans_to_list import read_jsonl
import re

def add_context_to_step(d, j):
    '''
    :param d: dict of steps per goal,
    :param j: existing json.
    :return:
    '''
    goal, str_step_num_start1 = j["id"].split("||")
    step_num = int(str_step_num_start1) - 1
    context = d[goal][:step_num]
    j["context"] = context
    j["future_context"] = d[goal][step_num + 1:]
    return j


def process_states(answer_list_dicts, state='before'):
    str_answer = ''
    for ans_dict in answer_list_dicts:
        if ans_dict is not None:
            entity = ans_dict['entity']
            attribute = ans_dict['attribute']
            before = ans_dict['before']
            after = ans_dict['after']
            if state == 'before':
                add_str = f'{before} before. '
            else:
                add_str = f'{after} after. '
            curr_init_state = f'{attribute} of {entity} was {add_str} '
            str_answer += curr_init_state
    return str_answer

def ans_to_prepost_tuple(sc_answer: str, sc_patterns = [
    re.compile(p, re.IGNORECASE) for p in [
        "(.*) of (.*) was (.*) before and (.*) afterwards",
        "(.*) of (.*) were (.*) before and (.*) afterwards",
    ]
], sc_patterns_backup = [
    re.compile(p, re.IGNORECASE) for p in [
        "(.*) was (.*) before and (.*) afterwards",
        "(.*) were (.*) before and (.*) afterwards",
    ]
]):
    # assuming preposcond is in one of the following formats:
    # there are buggy candidates like this one:
    # ['readiness of ingredients was unprepared before and prepared afterwards',
    # 'location of the ingredients was else where before and gathered in on place afterwards',
    # 'location of mango was in the fridge before and on the counter afterwards',
    # 'ownership of mango was the property of the store before and the property of the shopper afterwards',
    # 'location of knife was in the drawer before and on the counter afterwards',
    # 'location of measuring cup was in the cupboard before and on the counter afterwards',
    ## 'size of mango',
    ## 'lemon and garlic was whole before and cut into pieces afterwards',
    # 'state of the ingredients was unprepared before and prepared afterwards']

    # sc_patterns = [
    #     re.compile(p, re.IGNORECASE) for p in [
    #         "(.*) of (.*) was (.*) before and (.*) afterwards",
    #         "(.*) of (.*) were (.*) before and (.*) afterwards",
    #     ]
    # ]
    for p in sc_patterns:
        for f in p.findall(sc_answer):
            attr = f[0]
            entity = f[1]
            before = f[2]
            after = f[3]
            return {'attribute': attr,
                    'entity': entity,
                    'before': before,
                    'after': after}
    # if no match.
    for p in sc_patterns_backup:
        for f in p.findall(sc_answer):
            attr = "state"
            entity = f[0]
            before = f[1]
            after = f[2]
            return {'attribute': attr,
                    'entity': entity,
                    'before': before,
                    'after': after}
    # if no match2.
    return None

# {
#    "id":"www.wikihow.com/Make-a-Bead-Ring||6",
#    "question":"Cut a strip of elastic. Slip two larger beads at either side of the small bead. Slip a small bead on one tail. Continue this pattern until you have enough beads to fit around the finger. Finish the ring by slipping the loose tails through the first small bead you placed on the elastic. Finished. Now, what happens?",
#    "answers":[
#       "focus of you was focused on making a bracelet before and admiring your bracelet afterwards"
#    ],
#    "question_metadata":{
#       "url":"www.wikihow.com/Make-a-Bead-Ring",
#       "step_id":"6",
#       "context":"Cut a strip of elastic. Slip two larger beads at either side of the small bead. Slip a small bead on one tail. Continue this pattern until you have enough beads to fit around the finger. Finish the ring by slipping the loose tails through the first small bead you placed on the elastic.",
#       "query":"Finished.",
#       "future_context":"",
#       "topic":"Hobbies and Crafts",
#       "image_url":"https://www.wikihow.com/images/thumb/f/f6/Make-a-Bead-Ring-Step-6.jpg/aid63973-v4-728px-Make-a-Bead-Ring-Step-6.jpg"
#    },
#    "answers_metadata":[
#       {
#          "answer":"focus of you was focused on making a bracelet before and admiring your bracelet afterwards",
#          "entity":"you",
#          "before":"focused on making bracelet",
#          "after":"admiring bracelet",
#          "attr":"focus",
#          "modality":"with_image"
#       }
#    ],
#    "answers_sim":[
#
#    ]
# }

def generate_ques_json(id_, q):
    steps = [x.strip()+"." for x in q.replace(" Now, what happens?", "").split(".") if x]
    return {
          "url": id_.split("|")[0],
          "step_id": id_.split("|")[-1].strip(),
          "context": ".".join(steps[:-1]),
          "query": steps[-1],
          "future_context": "",
          "topic": "",
          "image_url": ""
       }

def generate_ans_json(ans_arr):
    o = []
    for ans in ans_arr:
            #"answer":"focus of you was focused on making a bracelet before and admiring your bracelet afterwards",
# #          "entity":"you",
# #          "before":"focused on making bracelet",
# #          "after":"admiring bracelet",
# #          "attr":"focus",
# #          "modality":"with_image"
        a={}
        a["answer"] = ans
        tup = ans_to_prepost_tuple(ans)
        if tup:
            a["attr"] = tup["attribute"]
            a["entity"] = tup["entity"]
            a["before"] = tup["before"]
            a["after"] = tup["after"]
        else:
            print(f"Skipping wrong answer format: {ans}")
            continue
        a["modality"] = "without_image"
        o.append(a)
        # we don't have this info because these 263 mismatching entries have url differences such as:
        # www.wikihow.com/Make-Four-legged-Animals-(Pipe-Cleaner-Crafts) vs. www.wikihow.com/Make-Four-Legged-Pipe-Cleaner-Animals
    return o

# Patch for https://github.com/allenai/openpi-dataset/issues/8
def patch_issue_8():
    od = dict()
    for partition in ["train", "dev", "test"]:
        ctr = 0
        old_lines = read_jsonl(f"data/formatted_for_gpt2/{partition}.jsonl")
        new_lines = read_jsonl(f"data/gold_v0/{partition}/id_answers.jsonl")
        for o, n in zip(old_lines, new_lines):
            assert o["id"] == n["id"], f"ID mismatch at: \n{o} \n and \n {n}"
            o_answers = [x.strip().replace(" were ", " was ").replace("was was", "was").replace("There will be no change.", "")  for x in o["answer"].split("afterwards,")]
            o_answers = [(x + " afterwards" if not x.endswith("afterwards") else x) for x in o_answers if x]
            n_answers = n["answers"]
            # if not n_answers and o_answers != n_answers:
            if o_answers != n_answers:
                ctr += 1
                # if "eaner" in n["id"] and "ipe" in n["id"]:
                #     print("debugging... ")
                o_copied = dict()
                # missed_answers = [x for x in set(o_answers) - set(n_answers)]
                o_copied["answers"] = o_answers
                o_copied["answers_metadata"] = generate_ans_json(o_answers)
                o_copied["question_metadata"] = generate_ques_json(id_=o["id"], q=o["question"])
                od[o["id"]] = o_copied
        print(f"{partition} has {ctr} answers replaced")
    return od

def tes_patch():
    for partition in ["train", "dev", "test"]:
        old_lines = read_jsonl(f"data/formatted_for_gpt2/{partition}.jsonl")
        new_lines = read_jsonl(f"data/gold_combined_and_clustered/{partition}.jsonl")
        for o, n in zip(old_lines, new_lines):
            assert o["id"] == n["id"], f"ID mismatch at: \n{o} \n and \n {n}"
            answers_old = [x.strip().replace(" were ", " was ").replace("was was", "was").replace("There will be no change.", "")  for x in o["answer"].split("afterwards,")]
            answers_old = [(x + " afterwards" if not x.endswith("afterwards") else x) for x in answers_old if x]
            answers_new = n["answers"]
            if len(answers_old) != len(answers_new) or answers_old != answers_new:
                print(f"\n\nmismatch ({partition}) {o['id']} \n"
                      # f"old:\n{answers_old} \nvs new:\n{answers_new} at\n\n "
                      # f"{[(x,y) for x,y in zip(answers_old, answers_new) if x!=y]}")
                      f"{set(answers_old) - set(answers_new)}")

    for in_fp in ["data/gold_combined_and_clustered/train.jsonl",
                  "data/formatted_for_gpt2/train.jsonl",
                  "data/gold_combined_and_clustered/dev.jsonl",
                  "data/formatted_for_gpt2/dev.jsonl",
                  "data/gold_combined_and_clustered/test.jsonl",
                  "data/formatted_for_gpt2/test.jsonl"
                  ]:
        with open(in_fp) as infile:
            cnt_empty = 0
            for line in infile:
                j = json.loads(line)
                if "answers" in j:
                    answers = j["answers"]
                else:
                    answers = [x.strip().replace(" were ", " was ").replace("was was", "was").replace("There will be no change.", "")  for x in j["answer"].split("afterwards,")]
                    answers = [(x + " afterwards" if not x.endswith("afterwards") else x) for x in answers if x]

                if len(answers) == 0:
                        cnt_empty += 1
        print(f"{in_fp} has {cnt_empty} empty answer entries.")


if __name__ == '__main__':
    # od=patch_issue_8()
    tes_patch()


# Tricky before/after examples (incomplete after if split in metadata)
#                              (knowledge of x was less about the fact that y is important)
# 'knowledge of you was less aware light is important before and important afterwards'