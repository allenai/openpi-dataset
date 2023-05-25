''' 
script to retrieve the original data entries corresponding to the mini dataset
'''

import json 


def main():
    mini_data = json.load(open('../data/dev-data-mini.json', 'r')) 
    original_data = json.load(open('../../../../data/dev-ranked.json', 'r'))    

    mini_data_keys = list(mini_data.keys())
    out_data = {}
    for key in original_data.keys():
        if key in mini_data_keys:
            out_data[key] = original_data[key]
    
    json.dump(out_data, open('../data/dev-data-mini-original.json', 'w'), indent=4)


if __name__ == '__main__':
    main()