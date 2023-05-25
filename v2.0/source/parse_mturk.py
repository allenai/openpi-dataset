# Script to parse outdated mturk data

import csv
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

def parse_state_change(state_change):
    attribute = state_change.split(" of ")[0]
    if " was " in state_change:
        state_change = state_change.replace(" was ", "@@@", 1)
    elif " were " in state_change:
        state_change = state_change.replace(" were ", "@@@", 1)
    entity = state_change.split(" of ")[1].split("@@@")[0]
    before = state_change.split("@@@")[1].split(" before ")[0]
    after = state_change.split("@@@")[1].split(" and ")[1].split(" afterwards")[0]
    return entity, attribute, before, after
    

proc_step_entities = {}
proc_step_attributes = {}
with open("../data/xx_json_dev_without_image.jsonl") as f:
    for line in f:
        row = json.loads(line)
        step_entities = []
        step_attributes = []
        step_conditions = []
        title = row["procedure_title"]
        if title not in proc_step_entities:
            proc_step_entities[title] = {}
            proc_step_attributes[title] = {}
        step = row["step"]
        if step not in proc_step_entities[title]:
            proc_step_entities[title][step] = {}
            proc_step_attributes[title][step] = {}
        response = row["state_change"]
        entity, attribute, before, after = parse_state_change(response)
        num_for_votes = row["num_for_votes"]
        num_against_votes = row["num_against_votes"]
        if entity not in proc_step_entities[title][step]:
            proc_step_entities[title][step][entity] = [0,0]
        if attribute not in proc_step_attributes[title][step]:
            proc_step_attributes[title][step][attribute] = [0,0]
        proc_step_entities[title][step][entity][0] += num_for_votes
        proc_step_entities[title][step][entity][1] += num_against_votes
        proc_step_attributes[title][step][attribute][0] += num_for_votes
        proc_step_attributes[title][step][attribute][1] += num_against_votes

proc_entities = {}
proc_attributes = {}
for title, a in proc_step_entities.items():
    if title not in proc_entities:
        proc_entities[title] = {}
    for step, b in a.items():
        for entity, (num_for_votes, num_against_votes) in b.items():
            if entity not in proc_entities[title]:
                proc_entities[title][entity] = [num_for_votes, num_against_votes]
            else:
                proc_entities[title][entity][0] += num_for_votes
                proc_entities[title][entity][1] += num_against_votes

for title, a in proc_step_attributes.items():
    if title not in proc_attributes:
        proc_attributes[title] = {}
    for step, b in a.items():
        for attributes, (num_for_votes, num_against_votes) in b.items():
            if attributes not in proc_attributes[title]:
                proc_attributes[title][attributes] = [num_for_votes, num_against_votes]
            else:
                proc_attributes[title][attributes][0] += num_for_votes
                proc_attributes[title][attributes][1] += num_against_votes

with open("../data/proc_step_entities.json", "w") as f:
    json.dump(proc_step_entities, f, indent=4)
with open("../data/proc_step_attributes.json", "w") as f:
    json.dump(proc_step_attributes, f, indent=4)
with open("../data/proc_entities.json", "w") as f:
    json.dump(proc_entities, f, indent=4)
with open("../data/proc_attributes.json", "w") as f:
    json.dump(proc_attributes, f, indent=4)