'''
script to sample a small subset of the dev data for testing purposes (10 data points)
'''

import os
import json
import pickle
import numpy as np


def main():
    with open('../../../data/dev-ranked.json', 'r') as f:
        data = json.load(f)
    f.close()

    selected_idx = np.random.choice(list(data.keys()), 10, replace=False)
    out_data = {key: val for (key, val) in data.items() if key in selected_idx} 

    with open('./dev-data-mini.json', 'w') as f:
        json.dump(out_data, f, indent=4)    
    f.close()


if __name__ == '__main__':
    main()