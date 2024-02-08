# An API that takes in a procedure and predicts
# - the schema (entities and attributes that undergo changes)
# - the values of these changes
# - the salience of these entities

import json
import random
import openai
import backoff
import argparse
import json
import re

random.seed(299)

parser = argparse.ArgumentParser()
parser.add_argument('--key', default='harry_ccbft', type=str, help='The name of the OpenAI API key file.')
parser.add_argument('--input', required=True, type=str, help='Path to the input file.')
parser.add_argument('--output', required=True, type=str, help='Path to the output file.')
parser.add_argument('--gs', action="store_true", help='Whether to do global salience prediction.')
parser.add_argument('--ls', action="store_true", help='Whether to do local salience prediction.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

SCHEMA_INSTRUCTIONS = """For each of the following steps, list the involved entities and attributes that have changed in a JSON format. Keep in mind of the following
1. Do not list those that do not change. For example, for the step 'heat the oven', rack (temperature) is correct, while oven (color) is wrong. 
2. If there are more than one instance of an entity, you must clearly distinguish them. For example, if there are two eggs, you can write 'egg 1' and 'egg 2'. 
3. If the same entity is mentioned with different names, you must keep their names consistent. For example, you should name both 'glass' and 'wine glass' as 'glass', if they refer to the same thing.
Is this clear?"""

STATE_INSTRUCTIONS = """For each combination of an entity and an attribute, provide its state changes. Your answer will include the before and after states in a JSON format. Pay attention to the following:
1. If a state is unclear, mark it as 'unknown'.
2. If an entity is nonexistent before or after the step, mark the state as 'nonexistent'.
Are you ready?"""

def apply_fewshot_template(examples):
    template = [
        {"role": "user", "content": SCHEMA_INSTRUCTIONS},
        {"role": "assistant", "content": "Yes, I understand."},
        {"role": "user", "content": STATE_INSTRUCTIONS},
        {"role": "assistant", "content": "Yes, I'm ready."},
    ]
    for example in examples:
        template.append({"role": "user", "content": f"Goal: {example['goal'].lower()}.\n" + '\n'.join(["Step " + str(id) + '. ' + s for id, s in enumerate(example['steps'])])})
        output_json = {}
        for i, (step, e_a) in enumerate(example["gold_step_entities_attributes"].items()):
            output_json[i] = {"step": example["steps"][i], "entities": {}}
            for entity, att_bef_afts in e_a.items():
                att_bef_afts = [{
                    "attribute": att_bef_aft[0],
                    "before": att_bef_aft[1],
                    "after": att_bef_aft[2],
                } for att_bef_aft in att_bef_afts]
                output_json[i]["entities"][entity] = att_bef_afts
        template.append({"role": "assistant", "content": json.dumps(output_json)})
    #print(template)
    #raise SystemExit
    return template

def apply_inference_template(example):
    return {"role": "user", "content": f"Goal: {example['goal'].lower()}.\n" + '\n'.join(["Step " + str(id) + '. ' + s for id, s in enumerate(example['steps'])])}

def build_fewshot():
    with open("one_shot.json", "r") as f:
        one_shot = json.load(f)

    selected_examples = [one_shot]
    fewshot = apply_fewshot_template(selected_examples)
    #print(fewshot)
    #raise SystemExit        
    return fewshot

# @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError, openai.error.APIConnectionError))
# def run_gpt(prompt, model="text-davinci-003", temperature=0, stop=['\n']):
#     ret = openai.Completion.create(
#         engine=model,
#         prompt=prompt,
#         temperature=temperature,
#         max_tokens=200,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=stop
#     )
#     gen_text = ret["choices"][0]["text"].strip()#.split('\n')[0]
#     return gen_text

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError, openai.error.APIConnectionError))
def run_chatgpt(prompt, model="gpt-3.5-turbo", temperature=0):
    ret = openai.ChatCompletion.create(
        model=model,
        messages=prompt
        )
    gen_text = dict(ret["choices"][0]["message"])["content"]
    return gen_text

def predict_states(example):
    prompt_fewshot = build_fewshot()
    apply_template = apply_inference_template
    run = run_chatgpt

    template = []
    template += prompt_fewshot

    prompt_inference = apply_template(example)
    template.append(prompt_inference)
    #print(template)
    #raise SystemExit
    output_str = run(template)
    #print(output_str)
    #raise SystemExit
    template.append({"role": "assistant", "content": output_str})
    output_json = json.loads(output_str)
    #output_json = [x["entities"] for x in output_json.values()]
    #print(output_json)
    #raise SystemExit

    return output_json

def parse_generated_text(gen_text):
    # Eraser: 5 - The eraser is the main component of the instruction and is essential for cleaning the inside of the windshield.
    #print(gen_text)
    try:
        score = re.search(r'\d+', gen_text).group()
        explanation = gen_text
    except:
        score = 1
        explanation = ""
        print("Error parsing generated text.", gen_text)
    return score, explanation

def predict_global_salience(goal, steps, entities):
    prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
               {"role": "user", "content": f"Here are some instructions on \"{goal}\".\n" + '\n'.join(['- ' + s for s in steps]) + "\n" + "Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance in the instruction. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
    prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
    output = []
    for entity in entities:
        prompt.append({"role": "user", "content": entity})
        gen_text = run_chatgpt(prompt)
        score, explanation = parse_generated_text(gen_text)
        output.append({"entity": entity, "global_salience_pred": int(score), "global_salience_explanation": explanation})
        prompt.append({"role": "assistant", "content": gen_text})
    return sorted(output, key=lambda d: d['global_salience_pred'], reverse=True) 

def predict_local_salience(goal, steps, entities):
    local_output = [[] for x in range(len(steps))]
    for i, step in enumerate(steps):
        prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
                {"role": "user", "content": f"One of the step of \"{goal}\" is \"{step}\". Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance for finishing this step. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
        prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
        for entity in entities:
            prompt.append({"role": "user", "content": entity})
            gen_text = run_chatgpt(prompt)
            score, explanation = parse_generated_text(gen_text)
            local_output[i].append({"entity": entity, "local_salience_pred": int(score), "local_salience_explanation": explanation})
            prompt.append({"role": "assistant", "content": gen_text})
        local_output[i] = sorted(local_output[i], key=lambda d: d['local_salience_pred'], reverse=True) 
    #print(local_output)
    return local_output

if __name__ == "__main__":
    with open(args.input, "r") as f:
        examples = json.load(f)["input"]
    out_dict = {}
    for i, example in enumerate(examples):
        print(i)
        id = example["id"]
        out_dict[id] = {}
        goal = example["goal"]
        steps = example["steps"]
        step_states = predict_states(example)
        out_dict[id]["states"] = step_states.copy()
        #print(step_states)
        #raise SystemExit

        if args.gs or args.ls:
            all_entities = [[ent for ent in s["entities"].keys()] for s in step_states.values()]
            all_entities = [item for sublist in all_entities for item in sublist]
            all_entities = list(set(all_entities))
            print(all_entities)

        if args.gs:
            global_output = predict_global_salience(goal, steps, all_entities)
            print(global_output)
            out_dict[id]["global_salience"] = global_output.copy()
        if args.ls:
            local_output = predict_local_salience(goal, steps, all_entities)
            #print(local_output)
            out_dict[id]["local_salience"] = local_output.copy()

    with open(args.output, "w") as f:
        json.dump(out_dict, f, indent=2)