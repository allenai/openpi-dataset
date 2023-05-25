import os
import ast
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_txt, save_txt

from transformers import GPT2Tokenizer


def get_input_token_count(input: str, tokenizer: callable) -> int:
    return len(tokenizer(input).input_ids)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--template_path', type=str, required=True, help='Path to template folder'
    )
    return parser.parse_args()


def load_templates() -> str:
    header_template = load_txt(os.path.join(args.template_path, 'chat_header.txt'))
    content_template = load_txt(os.path.join(args.template_path, 'chat_content.txt'))
    response_template = load_txt(os.path.join(args.template_path, 'chat_response.txt'))
    return header_template, content_template, response_template


def load_formulas() -> callable:
    procedure_context_formula = lambda goal, steps: f"I am trying to {goal}. Here are the steps. {steps} Do you get it?"
    entity_context_formula = lambda narrative: f"Here is a description of the state change of an object involved in the procedure: {narrative}"
    return procedure_context_formula, entity_context_formula


def clean_content(content: str) -> str:
    return content.split(':')[-1].strip()
    

def main():
    global args
    args = get_args()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    np.random.seed(42)

    gpt3_examples = load_txt('./assets/examples/example1.txt')
    gpt3_example_lst = gpt3_examples.strip().split('\n\n\n')

    header_template, content_template, response_template = load_templates()
    procedure_formula, entity_formula = load_formulas()

    header = ast.literal_eval(header_template)
    out_example = [header]
    token_count = get_input_token_count(header['content'], tokenizer)

    MAX_TOKEN = 2500
    while token_count < MAX_TOKEN:

        sample_idx = np.random.choice(len(gpt3_example_lst), 1)[0]
        content = gpt3_example_lst.pop(sample_idx)

        context, answer = content.split('\n\n')
        goal, steps, entity_state = context.strip().split('\n')
        goal, steps, entity_state = clean_content(goal), clean_content(steps), clean_content(entity_state)

        # cleanup answer
        answer = "Here are the alternative ways to describe this object state:" + answer.split(':')[1:][0]

        # make current content template
        cur_content = content_template.replace('{procedure_context}', procedure_formula(goal.lower(), steps))  \
                                      .replace('{entity_context}', entity_formula(entity_state))
        cur_answer =  response_template.replace('{answer}', answer)

        cur_content = ast.literal_eval(cur_content)
        cur_answer = [ast.literal_eval(cur_answer.strip().replace('\n', '\\n'))]

        out_example += cur_content
        out_example += cur_answer

        token_count += sum([get_input_token_count(entry['content'], tokenizer) for entry in cur_content])
        token_count += get_input_token_count(cur_answer[0]['content'], tokenizer)

    with open('./assets/examples/dev-ranked-declustered-chat-examples.pkl', 'wb') as f:
        pickle.dump(out_example, f)
    f.close()


if __name__ == '__main__':
    main()


