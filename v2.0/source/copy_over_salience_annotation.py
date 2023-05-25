import json

with open("../data/dev-data-reformatted-v4_salience.json") as f1, open("../data/dev-data-reformatted-v4_votes_salience1.json") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)
    for (id1, a1), (id2, a2) in zip(data1.items(), data2.items()):
        #print(id1)
        #assert id1 == id2
        #if int(id1) > 13:
        #    break
        if int(id1) < 14:
            continue
        if int(id1) > 20:
            break
        print(id1)
        #raise SystemExit
        entity_annotations = {}
        for entitiy, content in a1["clusters"].items():
            #print(content)
            entity_annotations[entitiy] = {"global_salience": content["global_salience"], "local_salience": content["local_salience"]}

        entity_to_names = a2["clusters"]

        for i, entity_state in enumerate(a2["states"]):
            for name in entity_to_names[entity_state["entity"]]["entity_cluster"]:
                #print(entity_annotations)
                #raise SystemExit
                if name in entity_annotations:
                    #print(name)
                    #raise SystemExit
                    data2[id2]["states"][i]["global_salience"] = entity_annotations[name]["global_salience"]
                    for (j, b), local_salience in zip(entity_state["answers"].items(), entity_annotations[name]["local_salience"]):
                        data2[id2]["states"][i]["answers"][j]["local_salience"] = local_salience
                    break

with open("../data/dev-data-reformatted-v4_votes_salience2.json", "w") as f:
    json.dump(data2, f, indent=4)