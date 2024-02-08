import random
random.seed(299)
import json
import argparse
import re

from tqdm import tqdm
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer, 
    GenerationConfig,
    StoppingCriteria, 
    StoppingCriteriaList,
)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer: LlamaTokenizer, stops = [], encounters: int =1, device: str = 'cuda'):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False


class Llama2Chat():

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'meta-llama/Llama-2-70b-chat-hf'
    WEIGHT_DIR = '/nlp/data/huggingface_cache/models--meta-llama--Llama-2-70b-chat-hf/snapshots/36d9a7388cc80e5f4b3e9701ca2f250d21a96c30/'
    CACHE_DIR = '/nlp/data/huggingface_cache/'
    generation_config = None
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(self):
        self._init_tokenizer()
        self._init_model()

    def _init_tokenizer(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            'meta-llama/Llama-2-70b-chat-hf',
            cache_dir = self.CACHE_DIR,
            key='~/hainiu_hf_token.key'
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _init_model(self):
        # NOTE: quantize the model to fp16
        self.model = LlamaForCausalLM.from_pretrained(
            self.WEIGHT_DIR,
            device_map="auto", 
            torch_dtype = torch.float16,
        )
        self.model = torch.compile(self.model)

    def _tokenize_prompt(self, prompt: str) -> dict:
        return self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda:0')

    def _convert_to_llama_prompt(self, chatgpt_prompt: list) -> str:
        llama_prompt = ""
        llama_prompt += f"<s>{self.B_INST} {self.B_SYS}\n{chatgpt_prompt[0]['content']}\n{self.E_SYS}\n"
        for idx, content_dict in enumerate(chatgpt_prompt):
            # skip system prompt
            if idx == 0:
                continue
            elif idx == 1:
                llama_prompt += f"{content_dict['content']} {self.E_INST}"
            else:
                if content_dict['role'] == 'user':
                    llama_prompt += f"<s> {self.B_INST}{content_dict['content']} {self.E_INST}"
                elif content_dict['role'] == 'assistant':
                    llama_prompt += f"{content_dict['content']} </s>"

        return llama_prompt

    def inference(self, prompt: str):
        llama_prompt = self._convert_to_llama_prompt(prompt)
        input = self._tokenize_prompt(llama_prompt)
        output = self.model.generate(**input).squeeze()
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        output = output.split(self.E_INST)[-1].strip()

        return output


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt-3.5-turbo', type=str, help='Model name.')
parser.add_argument('--key', default='harry_ccbft', type=str, help='The name of the OpenAI API key file.')
parser.add_argument('--seed', default='', type=str, help='Random seed.')
parser.add_argument('--split', default='dev', type=str, help='The split to evaluate on.')

args = parser.parse_args()
if args.seed:
    random.seed(int(args.seed[1:]))


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

def predict_global(goal, steps, entities, pbar):
    prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
               {"role": "user", "content": f"Here are some instructions on \"{goal}\".\n" + '\n'.join(['- ' + s for s in steps]) + "\n" + "Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance in the instruction. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
    prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
    output = {}
    for entity in entities:
        prompt.append({"role": "user", "content": entity})
        gen_text = llama2chat.inference(prompt)
        score, explanation = parse_generated_text(gen_text)
        output[entity] = {"global_salience_pred": score, "global_salience_explanation": explanation}
        prompt.append({"role": "assistant", "content": gen_text})
        pbar.update(1)
    return output

def predict_local(goal, steps, entities, pbar):
    local_output = [{} for x in range(len(steps))]
    for i, step in enumerate(steps):
        prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
                {"role": "user", "content": f"One of the step of \"{goal}\" is \"{step}\". Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance for finishing this step. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
        prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
        for entity in entities:
            prompt.append({"role": "user", "content": entity})
            gen_text = llama2chat.inference(prompt)
            score, explanation = parse_generated_text(gen_text)
            local_output[i][entity] = {"local_salience_pred": score, "local_salience_explanation": explanation}
            prompt.append({"role": "assistant", "content": gen_text})
            pbar.update(1)
    #print(local_output)
    return local_output


llama2chat = Llama2Chat()  
with open("../data/dev-data-reformatted-v4.json") as f:
    data = json.load(f)
    for id, a in tqdm(data.items(), position=0, leave=False):
        goal = a["goal"]
        steps = a["steps"]
        entities = [state["entity"] for state in a["states"]]
        with tqdm(range(len(steps) + (len(steps) * len(entities))), position=1, leave=False) as pbar:
            global_output = predict_global(goal, steps, entities, pbar)
            local_output = predict_local(goal, steps, entities, pbar)
        for i, state in enumerate(a["states"]):
            #print(global_output[state["entity"]])
            data[id]["states"][i].update(global_output[state["entity"]])
            for j, step_num in enumerate(a["states"][i]["answers"]):
                data[id]["states"][i]["answers"][step_num] = {"attributes": data[id]["states"][i]["answers"][step_num]}
                data[id]["states"][i]["answers"][step_num].update(local_output[j][state["entity"]])
        #if id == "20":
        #    break

with open("../data/dev-data-reformatted-v4_pred-salience-llama2-70.json", "w") as f_out:
    json.dump(data, f_out, indent=4)
