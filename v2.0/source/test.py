import json


# NOTE: The script now generate device map that uses 8 GPUs
# NOTE: The first GPU is reserved for storing the input, the rest of them are used for loading the model
def main():
    with open('./llama-65b-device-map.json', 'r') as f:
        map = json.load(f)
    f.close()

    block = len(map) // 7
    
    for idx, key in enumerate(map.keys()):
        cur_idx =  idx // block + 1
        if cur_idx <= 7:
            map[key] = cur_idx
        else:
            map[key] = 7

    with open('./llama-65b-device-map.json', 'w') as f:
        json.dump(map, f, indent=4)
    f.close()


if __name__ == "__main__":
    main()
