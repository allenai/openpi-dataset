import os
import re
import ast
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from transformers import GPT2Tokenizer

from openai_inference import gpt3_inference, chat_inference
from utils import load_json, save_json, load_template, load_txt


def get_input_token_count(input: str, tokenizer: callable) -> int:
    return len(tokenizer(input).input_ids)


def load_templates() -> str:
    header_template = load_txt(os.path.join(args.template_path, 'chat_header.txt'))
    content_template = load_txt(os.path.join(args.template_path, 'chat_content.txt'))
    response_template = load_txt(os.path.join(args.template_path, 'chat_response.txt'))
    return header_template, content_template, response_template


def make_example(example_candidates: list, tokenizer: callable):
    MAX_TOKEN = 2500 

    out_example = ''
    total_token = 0
    while total_token < MAX_TOKEN:
        sample_idx = np.random.choice(len(example_candidates), 1)[0]
        cur_example = example_candidates.pop(sample_idx).strip()
        out_example += cur_example + '\n\n\n'
        total_token += get_input_token_count(cur_example + '\n\n\n', tokenizer)
    
    return out_example


def load_formulas() -> callable:
    procedure_context_formula = lambda goal, steps: f"I am trying to {goal}. Here are the steps. {steps} Do you get it?"
    entity_context_formula = lambda narrative: f"Here is a description of the state change of an object involved in the procedure: {narrative}"
    return procedure_context_formula, entity_context_formula


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the openpi dataset'
    )
    parser.add_argument(
        '--template_path', type=str, required=True, help='path to the prompt template'
    )
    parser.add_argument(
        '--model_name', type=str, default='gpt-3.5-turbo', help='model name from OpenAI API'
    )
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the file that stores OpenAI API key'
    )
    parser.add_argument(
        '--examples_path', type=str, required=True, help='path to in-context examples'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to save the gpt3 output'
    )
    parser.add_argument(
        '--zero_shot', type=int, default=0, help='whether to use zero-shot learning'
    )
    return parser.parse_args()


def main():
    np.random.seed(42)
    global args
    args = get_args()
    data = load_json(args.data_path)

    # regular expression to extract keywords
    extract_template = re.compile('The (.*) of (.*) is (.*) before and (.*) afterwards.')

    openai.api_key_path = args.api_path

    if 'turbo' not in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        template = load_txt(args.template_path)

        # process examples to fit context window
        examples = load_txt(args.examples_path)
        example_candidates = examples.strip().split('\n\n\n')
        examples = make_example(example_candidates, tokenizer)

    else:
        header, content, _ = load_templates()
        procedure_context_formula, entity_context_formula = load_formulas()
        with open(args.examples_path, 'rb') as f:
            examples = pickle.load(f)
        f.close()

    for counter, (key, entry) in enumerate(tqdm(data.items(), position=0, leave=False)):
        if counter <= 50:
            continue
            
        # if counter > 50:
        #     break

        goal, steps, entity_state_lst = entry.values()
        steps = ' '.join(steps)

        result_dict = {}
        for i, entity_state in enumerate(tqdm(entity_state_lst, position=1, leave=False)):
            gpt3_output = []
            cur_entity = entity_state['entity']
            for narrative in entity_state['openpi_narrative']:
                if 'turbo' not in args.model_name:
                    cur_entry = template.replace('{goal}', goal.replace('"', "'"))  \
                                        .replace('{steps}', steps.replace('"', "'"))  \
                                        .replace('{openpi_template}', narrative.replace('"', "'"))
                else:
                    procedure_context = procedure_context_formula(
                        goal.lower().replace('"', "'"), 
                        steps.replace('"', "'")
                    )
                    entity_context = entity_context_formula(narrative.replace('"', "'"))

                    cur_content = content.replace('{procedure_context}', procedure_context)  \
                                         .replace('{entity_context}', entity_context)

                    cur_entry = [ast.literal_eval(header.strip())] +  \
                                 ast.literal_eval(cur_content)
                
                if args.zero_shot:
                    zero_shot_instruct = "Answer in the following format: 'The [attribute] of [entity] is [pre_condition] before and [post_condition] afterwards.'"
                    cur_entry[-1]['content'] += zero_shot_instruct
                else:
                    # add in-context example
                    cur_entry = examples + cur_entry

                if 'turbo' not in args.model_name:
                    out = gpt3_inference(cur_entry.strip(), args.model_name).strip()
                else:
                    out = chat_inference(cur_entry, args.model_name)['content']

                    if not args.zero_shot:
                        try:
                            out = out.split(':')[1].strip()
                        except:
                            out = 'None'

                out_components = extract_template.findall(out) 

                temp_component_lst = []
                for tup in out_components:
                    if len(tup) == 4:
                        attr, ent, pre, post = tup
                        temp_component_lst.append(
                            {
                                'attribute': attr,
                                'entity': ent,
                                'pre-state': pre,
                                'post-state': post
                            }
                        )

                gpt3_output.append(
                    {
                        'input': narrative,
                        'raw-output': out.strip().split('\n'),
                        'structured': temp_component_lst
                    }
                )

            result_dict[cur_entity] = gpt3_output

        data[key]['gpt3_output'] = result_dict
    save_json(args.out_path, data)

if __name__ == '__main__':
    main()

