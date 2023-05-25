"""
script to make in-context example foe attribute over-generation
"""

import json
import pickle
import argparse
import numpy as np
from copy import deepcopy


def save_pkl(data: list, path: str) -> None:
    """
    Function to save python list to pickle file

    Args:
        data: a python list object
        path: str path to save the pickle file

    Returns:
        None
    """

    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_json(path: str) -> dict:
    """
    Function to load json file as dictionary

    Args:
        path: str specifying the path to the json file

    Returns:
        a python dictionary object
    """

    data = json.load(open(path, 'r'))
    return data


def get_args() -> argparse.Namespace:
    """
    Get arguments from command line

    Returns:
        namespace containing all arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_proc', type=int, default=2, help='number of procedures used to make example'
    )
    return parser.parse_args()


def main():
    np.random.seed(42)
    args = get_args()
    data = load_json('../../../../data/data_in_new_format/dev-data-reformatted-v4.json')
    template = load_json('./attr_template.json')

    example_idx = list(np.random.choice(len(data), args.num_proc))
    save_pkl(example_idx, './attr_example_idx.pkl')

    sampled_data = {key: val for key, val in data.items() if int(key) in example_idx}
    
    output_example = []
    for proc_id, proc_content in sampled_data.items():
        cur_goal = proc_content['goal']
        cur_steps = proc_content['steps']
        cur_clusters = proc_content['clusters']
        
        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, I {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, I {step.lower()}')

        cur_goal = f'I am trying to {cur_goal.strip().lower()}. '
        cur_steps = cur_goal + cur_step_narratives[-1]

        # over generate the cluster entries
        for entity_id, entity_content in cur_clusters.items():
            cur_attr_cluster = entity_content['attribute_cluster']

            for attr_cluster in cur_attr_cluster.values():
                cur_template = deepcopy(template)

                # add procedure context
                cur_context = cur_template[0]['content'].replace('{context}', cur_steps) + ' Do you get the procedure?'
                cur_template[0]['content'] = cur_context

                # add entity context
                cur_attr_context = f'Here are some attributes that describe the same property of an object: {attr_cluster} of {entity_id}. What are some alternative names for this property? Organize them in a list. Answer \"None\" if there is no alternative name.'
                cur_template[2]['content'] = cur_template[2]['content'].replace('{attr_context}', cur_attr_context)

                # add answer
                cur_template[-1]['content'] = cur_template[-1]['content'].replace('{entity_ans}', '[]')

                output_example.extend(cur_template)

    # add header for ChatGPT API
    output_example = [{"role": "system", "content": "You are a helpful assistant."}] + output_example
    # with open('./attr_example.json', 'w') as f:
    #     json.dump(output_example, f, indent=4)
    # f.close()
    

if __name__ == "__main__":
    main()
