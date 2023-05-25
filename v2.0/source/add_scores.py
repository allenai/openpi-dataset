import json

with open("../data/dev-output.json") as f1, open("../data/dev-data-reformatted-v4_votes_salience_1-20.json") as f2:
    j1 = json.load(f1)
    j2 = json.load(f2)
    for (i, proc1), (i, proc2) in zip(j1.items(), j2.items()):
        mention_score_global = {}
        mention_score_local = {}
        for s1 in proc1["states"]:
            entity_mentions = s1["entity"].split(" | ")
            for m in entity_mentions:
                mention_score_global[m] = s1["saliency"]
                mention_score_local[m] = s1["step_saliency"]
        for j, s2 in enumerate(proc2["states"]):
            entity = s2["entity"]
            j2[i]["states"][j]["score"] = mention_score_global[entity]
            for (k, step), local_score in zip(s2["answers"].items(), mention_score_local[entity]):
                j2[i]["states"][j]["answers"][k]["score"] = local_score

with open("../data/dev-data-reformatted-v4_votes_score_salience_1-20.json", 'w') as f:
    json.dump(j2, f, indent=4)