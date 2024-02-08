import json

out_dict = {"input":[]}
with open("../data/dev-data-reformatted-v4.json") as f:
    data = json.load(f)
    count = 0
    for id, value in data.items():
        if count == 20:
            break
        out_dict["input"].append({"id": id, "goal": value["goal"], "steps": value["steps"]})
        count += 1

with open("example20_in.json", "w") as f:
    json.dump(out_dict, f, indent=4)