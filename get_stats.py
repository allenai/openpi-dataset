import json
from collections import defaultdict, Counter

topic_counter = defaultdict(set)
topic_state_change_counter = defaultdict(Counter)

for sp in ["train", "dev", "test"]:
    for modality in ["without", "with"]:
        fq_in = "./gold/gold_{}_image/{}/id_question_metadata.jsonl".format(modality, sp)
        fa_in = "./gold/gold_{}_image/{}/id_answers.jsonl".format(modality, sp)
        for a_meta, q_meta in zip(open(fa_in, 'r'), open(fq_in, 'r')):
            a_meta = json.loads(a_meta)
            q_meta = json.loads(q_meta)
            topic = q_meta["question_metadata"]["topic"]
            if topic != "unknown":
                url = q_meta["question_metadata"]["url"]
                num_state_changes = len(a_meta["answers"])
                topic_state_change_counter[topic][modality] += num_state_changes
                topic_counter[topic].add(url)
                topic_state_change_counter["All"][modality] += num_state_changes
                topic_counter["All"].add(url)
            else:
                pass

# if __name__ == '__main__':
    # from prettytable import PrettyTable
    # T = PrettyTable()
    # T.field_names = ["topic", "num articles", "y", "with-img", "wo-img"]
    # for k, v in topic_state_change_counter.items():
    #     num_articles = len(topic_counter[k])
    #     with_img = v["with"]
    #     without_img = v["without"]
    #     y = with_img + without_img
    #     T.add_row([k, num_articles, y, with_img, without_img])
    #
    # T.sortby = "num articles"
    # T.reversesort = True
    # print(T)

