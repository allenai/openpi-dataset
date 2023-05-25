import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--og', action='store_true', help='Whether to use overgenerated clusters.')
args = parser.parse_args()

if args.og:
    fname = "../data/dev-data-reformatted-v4-overgenerated-unique.json"
else:
    fname = "../data/dev-data-reformatted-v4.json"
with open(fname) as f_gold:
    gold_data = json.load(f_gold)

with open(f"../data/dev_schema_{args.model}_1.json") as f_pred:
    pred_data = json.load(f_pred)
    


def match_entity(entity, entity_canon_to_names):
    for canon, names in entity_canon_to_names.items():
        if entity in names:
            return canon
    return None

def match_attribute(entity, attribute, entity_canon_to_names, entity_canon_attribute_canon_to_names):
    for entity_canon, entity_names in entity_canon_to_names.items():
        if entity in entity_names:
            for attribute_canon, attribute_names in entity_canon_attribute_canon_to_names[entity_canon].items():
                if attribute in attribute_names:
                    return (entity_canon, attribute_canon)
    return None

def lemmatize(s):
    # Strip the, a, an and lowercase
    s = s.lower()
    if s.startswith("the "):
        s = s[4:]
    elif s.startswith("a "):
        s = s[2:]
    elif s.startswith("an "):
        s = s[3:]
    return s

entity_statistics = {
    "gold_total" : 0,
    "pred_total" : 0,
    "gold_correct": 0,
    "pred_correct": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0
}

entity_local_statistics = {
    "gold_total" : 0,
    "pred_total" : 0,
    "gold_correct": 0,
    "pred_correct": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0
}

attribute_statistics = {
    "gold_total" : 0,
    "pred_total" : 0,
    "gold_correct": 0,
    "pred_correct": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0
}

attribute_local_statistics = {
    "gold_total" : 0,
    "pred_total" : 0,
    "gold_correct": 0,
    "pred_correct": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0
}

for (id, gold_proc), (id, pred_proc) in zip(gold_data.items(), pred_data.items()):
    # Get gold clusters
    if args.og:
        entity_canon_to_names = {lemmatize(k):[lemmatize(x) for x in v["entity_cluster_combined"]] for k, v in gold_proc["clusters"].items()}
    else:
        entity_canon_to_names = {lemmatize(k):[lemmatize(x) for x in v["entity_cluster"]] for k, v in gold_proc["clusters"].items()}
    entity_canon_attribute_canon_to_names = {}
    for ec, block in gold_proc["clusters"].items():
        if args.og:
            block["attribute_cluster"] = {k: v["attribute_cluster_combined"] for k, v in block["attribute_cluster"].items()}
        ec = lemmatize(ec)
        entity_statistics["gold_total"] += 1
        entity_canon_attribute_canon_to_names[ec] = {}
        lemmatized_attribute_block = {lemmatize(k):[lemmatize(x) for x in v] for k, v in block["attribute_cluster"].items()}
        entity_canon_attribute_canon_to_names[ec].update(lemmatized_attribute_block)
        attribute_statistics["gold_total"] += len(block["attribute_cluster"])

        entity_matched = False
        for step_block_pred in pred_proc:
            for pred_entity, pred_attributes in step_block_pred.items():
                if pred_entity in entity_canon_to_names[ec]:
                    entity_matched = True
                    for canon_attribute, attribute_possible_names in block["attribute_cluster"].items():
                        for pred_attribute in pred_attributes:
                            if pred_attribute in attribute_possible_names:
                                attribute_statistics["gold_correct"] += 1
                                break
        if entity_matched:
            entity_statistics["gold_correct"] += 1

    appeared_entities = set()
    appeared_entity_attributes = set()

    # Take a pass in the pred
    for step_block in pred_proc:
        for pred_entity, pred_attributes in step_block.items():
            pred_entity = lemmatize(pred_entity)
            pred_attributes = [lemmatize(a) for a in pred_attributes]
            entity_local_statistics["pred_total"] += 1
            if pred_entity not in appeared_entities:
                entity_statistics["pred_total"] += 1
                
            matched_canon_entity = match_entity(pred_entity, entity_canon_to_names)
            if matched_canon_entity:
                entity_local_statistics["pred_correct"] += 1
                if pred_entity not in appeared_entities:
                    entity_statistics["pred_correct"] += 1

            for pred_attribute in pred_attributes:
                attribute_local_statistics["pred_total"] += 1
                if (pred_entity, pred_attribute) not in appeared_entity_attributes:
                    attribute_statistics["pred_total"] += 1
                matched_canon_entity_canon_attribute = match_attribute(pred_entity, pred_attribute, entity_canon_to_names, entity_canon_attribute_canon_to_names)
                if matched_canon_entity_canon_attribute:
                    attribute_local_statistics["pred_correct"] += 1
                    if matched_canon_entity_canon_attribute not in appeared_entity_attributes:
                        attribute_statistics["pred_correct"] += 1
                appeared_entity_attributes.add((pred_entity, pred_attribute))

            appeared_entities.add(pred_entity)
    #break

    # Take a pass in the gold to check local
    entity_already_correct = set()
    entity_attribute_already_correct = set()
    for entity_block in gold_proc["states"]:
        canon_entity = entity_block["entity"]
        entity_possible_names = entity_canon_to_names[lemmatize(canon_entity)]
        for step_block_gold, step_block_pred in zip(entity_block["answers"].values(), pred_proc):
            if step_block_gold:
                entity_local_statistics["gold_total"] += 1
                for pred_entity, pred_attributes in step_block_pred.items():
                    for attribute_block in step_block_gold:
                        attribute_local_statistics["gold_total"] += 1
                    if pred_entity in entity_possible_names:
                        entity_local_statistics["gold_correct"] += 1
                        for attribute_block in step_block_gold:
                            canon_attribute = attribute_block["attribute"]
                            attribute_possible_names = entity_canon_attribute_canon_to_names[lemmatize(canon_entity)][canon_attribute]
                            for pred_attribute in pred_attributes:
                                if pred_attribute in attribute_possible_names:
                                    attribute_local_statistics["gold_correct"] += 1

    #break


# Stats calculation
entity_statistics["precision"] = entity_statistics["pred_correct"] / entity_statistics["pred_total"]
entity_statistics["recall"] = entity_statistics["gold_correct"] / entity_statistics["gold_total"]
entity_statistics["f1"] = 2 * entity_statistics["precision"] * entity_statistics["recall"] / (entity_statistics["precision"] + entity_statistics["recall"])

attribute_statistics["precision"] = attribute_statistics["pred_correct"] / attribute_statistics["pred_total"]
attribute_statistics["recall"] = attribute_statistics["gold_correct"] / attribute_statistics["gold_total"]
attribute_statistics["f1"] = 2 * attribute_statistics["precision"] * attribute_statistics["recall"] / (attribute_statistics["precision"] + attribute_statistics["recall"])

entity_local_statistics["precision"] = entity_local_statistics["pred_correct"] / entity_local_statistics["pred_total"]
entity_local_statistics["recall"] = entity_local_statistics["gold_correct"] / entity_local_statistics["gold_total"]
entity_local_statistics["f1"] = 2 * entity_local_statistics["precision"] * entity_local_statistics["recall"] / (entity_local_statistics["precision"] + entity_local_statistics["recall"])

attribute_local_statistics["precision"] = attribute_local_statistics["pred_correct"] / attribute_local_statistics["pred_total"]
attribute_local_statistics["recall"] = attribute_local_statistics["gold_correct"] / attribute_local_statistics["gold_total"]
attribute_local_statistics["f1"] = 2 * attribute_local_statistics["precision"] * attribute_local_statistics["recall"] / (attribute_local_statistics["precision"] + attribute_local_statistics["recall"])

print("Entity Statistics:")
print(entity_statistics)
print("Attribute Statistics:")
print(attribute_statistics)
print("Entity Local Statistics:")
print(entity_local_statistics)
print("Attribute Local Statistics:")
print(attribute_local_statistics)
