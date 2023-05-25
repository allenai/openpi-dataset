import json
import pickle

def strip_entitiy(entity):
    entity = entity.lower()
    if entity.startswith("the "):
        return entity[4:]
    if entity.startswith("a "):
        return entity[2:]
    if entity.startswith("an "):
        return entity[3:]
    return entity

fname = "dev-data-reformatted-v4"

with open(f"../data/{fname}.json") as f:
    data = json.load(f)

with open("../data/proc_step_entities.json", "r") as f:
    proc_step_entities = json.load(f)

with open("../data/proc_entities.json", "r") as f:
    proc_entities = json.load(f)

with open("../data/proc_step_attributes.json", "r") as f:
    proc_step_attributes = json.load(f)

with open("../data/proc_attributes.json", "r") as f:
    proc_attributes = json.load(f)

for id, a in data.items():
    title = "How to " + a["goal"]
    global_entity_count = proc_entities[title]
    local_entity_count = proc_step_entities[title]
    global_attribute_count = proc_attributes[title]
    local_attribute_count = proc_step_attributes[title]

    entity_canon_to_names = {}
    for canon, names in a["clusters"].items():
        entity_canon_to_names[canon] = [strip_entitiy(name) for name in names["entity_cluster"]]
    entity_attribute_canon_to_names = {}
    for ent_canon, ent_names in a["clusters"].items():
        for att_canon, att_names in ent_names["attribute_cluster"].items():
            entity_attribute_canon_to_names[(ent_canon,att_canon)] = att_names
            #print((ent_canon,att_canon))
        
    for i, entity_state in enumerate(a["states"]):
        entity = entity_state["entity"]
        entity_names = entity_canon_to_names[entity]
        data[id]["states"][i]["votes"] = 0
        for ent, ct in global_entity_count.items():
            if ent in entity_names:
                # design choice: for minus against
                data[id]["states"][i]["votes"] += ct[0] - ct[1]
        entity_proc_annotators = []
        for (j, step), (step, entity_annotators), (step, attribute_annotators) in zip(entity_state["answers"].items(), local_entity_count.items(), local_attribute_count.items()):
            data[id]["states"][i]["answers"][j] = {
                "votes": 0,
                "attributes": data[id]["states"][i]["answers"][j]
            }
            for ent, ct in entity_annotators.items():
                if ent in entity_names:
                    data[id]["states"][i]["answers"][j]["votes"] += ct[0] - ct[1]

            # add attribute votes
            for k, att_block in enumerate(data[id]["states"][i]["answers"][j]["attributes"]):
                attribute = att_block["attribute"]
                #print(id, entity, attribute, j)
                #print(data[id]["states"][i]["answers"][j]["attributes"])
                attribute_names = entity_attribute_canon_to_names[(entity,attribute)]
                data[id]["states"][i]["answers"][j]["attributes"][k] = {
                    "votes": 0,
                    "states": data[id]["states"][i]["answers"][j]["attributes"][k]
                }
                for att, ct in attribute_annotators.items():
                    if att in attribute_names:
                        # design choice: for minus against
                        data[id]["states"][i]["answers"][j]["attributes"][k]["votes"] += ct[0] - ct[1]

                #if (entity, att) in 


with open(f"../data/{fname}_votes.json", 'w') as fw:
    json.dump(data, fw, indent=4)