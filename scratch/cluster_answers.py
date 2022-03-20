import json
import os
from itertools import groupby
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


class SCGrouper:

    def __init__(self):

        print(f"Initializing sentsim model ...", end=' ')
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        print(f"[done]")

    def sent_sim(self, sentences: List[str], threshold: float):
        paraphrases = util.paraphrase_mining(self.model, sentences, top_k=1)
        sims = []
        for paraphrase in paraphrases:
            score, i, j = paraphrase
            if round(score, 2) >= threshold:
                # print(f"{sentences[i]}\t{sentences[j]}\t{score:0.2f})")
                sims.append([sentences[i], sentences[j], score])
        return sims

def process_a_file(in_fp, out_fp, sim_obj: SCGrouper, sim_threshold=0.8):
    print(f"Processing sims in {in_fp} ...")
    with open(in_fp) as infile:
        with open(out_fp, 'w') as outfile:
            for line in tqdm(infile):
                j = json.loads(line)
                if j["answers"]:
                    j["answers_sim"] = sim_obj.sent_sim(j["answers"], threshold=sim_threshold)
                else:
                    j["answers_sim"] = []
                outfile.write(json.dumps(j))
                outfile.write('\n')


if __name__ == '__main__':
    in_dir = "data/gold_combined"
    out_dir = "data/gold_combined_and_clustered"
    sim_obj = SCGrouper()
    for p in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        process_a_file(in_fp=os.path.join(in_dir, p),
                       out_fp=os.path.join(out_dir, p),
                       sim_obj=sim_obj
                       )
    print(f"Output in {out_dir}")









#     # orig_id_meta_entry: Dict[str, Any]
#     def group(self, input_id: str, ques: str, arr_of_meta: List[Dict[str, str]]) -> List[str]:
#         """
#         :param arr_of_meta: (answers_metadata) array from the test/id_answers_metadata.jsonl
#                             from an item in this array, extract "attr" "entity" "before" "after"
#                             # TODO add an answer_idx key to every item.
#         :return:
#         """
#         # (attr1, ent1, xxx1, yyy1)
#         outputs = []
#         ans_map = {x['answer']:idx for idx, x in enumerate(arr_of_meta)}
#         groupfunc1 = lambda x: (x["attr"].split(" ")[0]) + (x["entity"].split(" ")[0])
#         groupfunc2 = lambda x: (x["attr"].split(" ")[0])
#         groupfunc = groupfunc2
#         result = []
#         for k, g in groupby(sorted(arr_of_meta, key=groupfunc), key=groupfunc):
#             result.append(list(g))
#         for cluster_idx, r in enumerate(result):
#             for a in r:
#                 assert a['answer'] in ans_map, f"{a['answer']} not found in:\n{a}"
#                 outputs.append(f"{input_id}\t{ques}\t{ans_map[a['answer']]}\tc{cluster_idx}\t{a['answer']}")
#         return outputs
#
#     def test_grouping(self):
#         answers = [
#             "flexibility of biscuits was pliable before and hard afterwards",
#             "location of biscuits was on counter before and in freezer afterwards",
#             "location of cookies was on table before and in freezer afterwards",
#             "placement of the bowl was outside of freezer before and now inside of freezer afterwards",
#             "temperature of biscuits was warm before and cold afterwards"
#             "temperature of biscuits was warm before and freezing afterwards",
#             "temperature of cookies was warm before and frozen afterwards",
#             "texture of cookies was soft before and hard afterwards",
#             "volume of freezer was empty before and full afterwards"
#         ]
#         j = json.loads('''
#         {"answers_metadata":[
#             {
#                 "answer":"location of cookies was on table before and in freezer afterwards",
#                 "entity":"cookies",
#                 "before":"on table",
#                 "after":"in freezer",
#                 "attr":"location",
#                 "modality":"with_image"
#             },
#             {
#                 "answer":"temperature of cookies was warm before and frozen afterwards",
#                 "entity":"cookies",
#                 "before":"warm",
#                 "after":"frozen",
#                 "attr":"temperature",
#                 "modality":"with_image"
#             },
#             {
#                 "answer":"texture of cookies was soft before and hard afterwards",
#                 "entity":"cookies",
#                 "before":"soft",
#                 "after":"hard",
#                 "attr":"texture",
#                 "modality":"with_image"
#             },
#             {
#                 "answer":"placement of the bowl was outside of freezer before and now inside of freezer afterwards",
#                 "entity":"the bowl",
#                 "before":"outside of freezer",
#                 "after":"now inside of freezer",
#                 "attr":"placement",
#                 "modality":"with_image"
#             },
#             {
#                 "answer":"temperature of biscuits was warm before and cold afterwards",
#                 "entity":"biscuits",
#                 "before":"warm",
#                 "after":"cold",
#                 "attr":"temperature",
#                 "modality":"without_image"
#             },
#             {
#                 "answer":"flexibility of biscuits was pliable before and hard afterwards",
#                 "entity":"biscuits",
#                 "before":"pliable",
#                 "after":"hard",
#                 "attr":"flexibility",
#                 "modality":"without_image"
#             },
#             {
#                 "answer":"volume of freezer was empty before and full afterwards",
#                 "entity":"freezer",
#                 "before":"empty",
#                 "after":"full",
#                 "attr":"volume",
#                 "modality":"without_image"
#             },
#             {
#                 "answer":"location of biscuits was on counter before and in freezer afterwards",
#                 "entity":"biscuits",
#                 "before":"on counter",
#                 "after":"in freezer",
#                 "attr":"location",
#                 "modality":"without_image"
#             },
#             {
#                 "answer":"temperature of biscuits was warm before and freezing afterwards",
#                 "entity":"biscuits",
#                 "before":"warm",
#                 "after":"freezing",
#                 "attr":"temperature",
#                 "modality":"without_image"
#             }
#         ]}''')
#         arr_of_meta = j["answers_metadata"]
#         ans_map = {x['answer']: idx for idx, x in enumerate(arr_of_meta)}
#         groupfunc1 = lambda x: (x["attr"].split(" ")[0]) + (x["entity"].split(" ")[0])
#         groupfunc2 = lambda x: (x["attr"].split(" ")[0])
#         groupfunc = groupfunc2
#         result = []
#         for k, g in groupby(sorted(arr_of_meta, key=groupfunc), key=groupfunc):
#             result.append(list(g))
#         return result
#
# #
# # if __name__ == '__main__':
# #     s = SCGrouper()
# #     x = s.test_grouping()
# #     print(x)
#
# # if __name__ == '__main__':
# #     o = SCGrouper()
# #     j = json.loads(
# #         '{"id": "www.wikihow.com/Make-Asparagus-in-Serrano-Ham||1", "answers_metadata": [{"answer": "power of oven was off before, and on afterwards.", "entity": "oven", "before": "off", "after": "on", "attr": "power", "modality": "with_image"}, {"answer": "temperature of oven was cold before, and hot afterwards.", "entity": "oven", "before": "cold", "after": "hot", "attr": "temperature", "modality": "with_image"}, {"answer": "power of display was off before, and on afterwards.", "entity": "display", "before": "off", "after": "on", "attr": "power", "modality": "with_image"}, {"answer": "temperature of oven racks was cold before, and hot afterwards.", "entity": "oven racks", "before": "cold", "after": "hot", "attr": "temperature", "modality": "with_image"}, {"answer": "color of heating element was black before, and red afterwards.", "entity": "heating element", "before": "black", "after": "red", "attr": "color", "modality": "with_image"}, {"answer": "pressure of stove buttons was unpressed before, and pressed afterwards.", "entity": "stove buttons", "before": "unpressed", "after": "pressed", "attr": "pressure", "modality": "with_image"}, {"answer": "state of oven was cold before, and hot afterwards.", "entity": "oven", "before": "cold", "after": "hot", "attr": "state", "modality": "with_image"}]}')
# #     input_ans_file = "trackworld/training_data/emnlp_2020/orig_tw_2020-v1/test/id_answers_metadata.jsonl"
# #     out_file = "/tmp/test.input_to_human.tsv"
# #
# #     query_map = {}
# #     with open(input_ques_file, 'r') as open_file:
# #         for line in open_file:
# #             j = json.loads(line)
# #             query_map[j['id']] = j['question_metadata']['query']
# #     with open(out_file, 'w') as open_out_file:
# #         open_out_file.write(f"id\tquery\tans_idx\tcluster_idx_to_fix\tans\n")
# #         with open(input_ans_file, 'r') as open_in_file:
# #             for line in open_in_file:
# #                 j = json.loads(line)
# #                 grouped = o.group(input_id=j['id'], ques=query_map[j['id']] , arr_of_meta=j["answers_metadata"])
# #                 for g in grouped:
# #                     open_out_file.write(g)
# #                     open_out_file.write("\n")
# #                 open_out_file.write("----\t----\t----\t----\t----\n")

def demo2():
    # Single list of sentences - Possible tens of thousands of sentences
    sentences = ['The cat sits outside',
                 'A man is playing guitar',
                 'I love pasta',
                 'The new movie is awesome',
                 'The cat plays in the garden',
                 'A woman watches TV',
                 'The new movie is so great',
                 'Do you like pizza?']

    sentences = [
        "flexibility of biscuits was pliable before and hard afterwards",
        "location of biscuits was on counter before and in freezer afterwards",
        "location of cookies was on table before and in freezer afterwards",
        "placement of the bowl was outside of freezer before and now inside of freezer afterwards",
        "temperature of biscuits was warm before and cold afterwards",
        "temperature of biscuits was warm before and freezing afterwards",
        "temperature of cookies was warm before and frozen afterwards",
        "texture of cookies was soft before and hard afterwards",
        "volume of freezer was empty before and full afterwards"
    ]

    sentences = [
        "flexibility of biscuits was pliable before and hard afterwards",
        "location of biscuits was on counter before and in freezer afterwards",
        "location of cookies was on table before and in freezer afterwards",
        "placement of the bowl was outside of freezer before and now inside of freezer afterwards",
        "temperature of biscuits was warm before and cold afterwards",
        "temperature of biscuits was warm before and freezing afterwards",
        "temperature of cookies was warm before and frozen afterwards",
        "state of cookies was warm before and frozen afterwards",
        "texture of cookies was soft before and hard afterwards",
        "volume of freezer was empty before and full afterwards"
    ]

    sentences1 = [
        "flexibility of biscuits",
        "location of biscuits",
        "location of cookies",
        "placement of the bowl",
        "temperature of biscuits",
        "temperature of biscuits",
        "temperature of cookies",
        "state of cookies was warm before and frozen afterwards",
        "texture of cookies",
        "volume of freezer"
    ]

    sentences1 = ['cleanness of mixing spoon',
             'composition of ingredients',
             'composition of ingredients',
             'composition of rum and coconut',
             'location of spoon',
             'state of ingredients',
             'state of mixture',
             'wetness of spoon'
             ]


    sentences = ['cleanness of mixing spoon was clean before and dirty afterwards',
                 'composition of ingredients was just put in pitcher before and mixed afterwards',
                 'composition of ingredients was separate before and mixed afterwards',
                 'composition of rum and coconut was separated before and mixed together afterwards',
                 'location of spoon was outside of pitcher before and inside pitcher afterwards',
                 'state of ingredients was unmixed before and mixed thoroughly afterwards',
                 'state of mixture was flat before and bubbly afterwards',
                 'wetness of spoon was dry before and wet afterwards'
                 ]

    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    paraphrases = util.paraphrase_mining(model, sentences, top_k=3)

    for paraphrase in paraphrases:
        score, i, j = paraphrase
        if round(score,2) >= 0.8:
            print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))


if __name__ == '__main__':
    demo2()