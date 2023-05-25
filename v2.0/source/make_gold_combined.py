import json
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--og', action='store_true', help='Whether to use overgenerated clusters.')
args = parser.parse_args()

out_dict = {}

if args.og:
    fname = "../data/dev-data-reformatted-v4-overgenerated-unique.json"
else:
    fname = "../data/dev-data-reformatted-v4.json"

with open(fname) as f:
    data = json.load(f)
    for id, proc in data.items():
        out_dict[id] = [[] for x in range(len(proc["steps"]))]
        if args.og:
            entity_canon_to_names = {k:v["entity_cluster_combined"] for k, v in proc["clusters"].items()}
        else:
            entity_canon_to_names = {k:v["entity_cluster"] for k, v in proc["clusters"].items()}
        entity_canon_attribute_canon_to_names = {}
        for entity, block in proc["clusters"].items():
            if args.og:
                block["attribute_cluster"] = {k: v["attribute_cluster_combined"] for k, v in block["attribute_cluster"].items()}
            entity_canon_attribute_canon_to_names[entity] = {}
            lemmatized_attribute_block = block["attribute_cluster"]
            entity_canon_attribute_canon_to_names[entity].update(lemmatized_attribute_block)
        for entity_block in proc["states"]:
            entity = entity_block["entity"]
            for i, (step_n, step_block) in enumerate(entity_block["answers"].items()):
                
                att_sents = []
                for att_block in step_block:
                    attribute = att_block["attribute"]
                    #print(attribute)
                    before = att_block["before"]
                    after = att_block["after"]
                    for entity_name in entity_canon_to_names[entity]:
                        #print(entity_canon_to_names[entity])
                        for attribute_name in entity_canon_attribute_canon_to_names[entity][attribute]:
                            sent = f"{attribute_name} of {entity_name} was {before.split(' | ')[0]} before and {after.split(' | ')[0]} after"
                            att_sents.append(sent)
                if att_sents:
                    out_dict[id][i].append(att_sents)
                #print(att_sents)
                #raise SystemExit
                #break
        #break

if args.og:
    fname = "../data/dev_schema_gold_2_og.json"
else:
    fname = "../data/dev_schema_gold_2.json"

with open(fname, "w") as f_gold:
    json.dump(out_dict, f_gold, indent=4)