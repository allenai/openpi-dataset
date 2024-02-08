import os 
import json
import torch
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    StoppingCriteria, 
    StoppingCriteriaList,
)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], device='cpu') -> None:
        super().__init__()
        self.stops = stops
        self.device = device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            stop = stop.to(self.device)
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True 
        return False


class LLaMA():

    save_dir = "/nlp/data/huggingface_cache"

    def __init__(self, model_name, device_map=None):
        self.model_name = f'huggyllama/{model_name}'
        self.device_map = device_map if device_map else 'auto'
        os.environ["HF_HOME"] = self.save_dir  # set env for NLPGPU cache
        
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_model(self) -> LlamaForCausalLM:
        if self.device_map != 'auto':
            with open(self.device_map, 'r') as f:
                self.device_map = json.load(f)
            f.close()

        return LlamaForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.save_dir, 
            torch_dtype=torch.float16, 
            device_map=self.device_map,
        )

    def _load_tokenizer(self) -> LlamaTokenizer:
        return LlamaTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.save_dir
        )

    @torch.no_grad()
    def inference(self, prompt: str, stop: list=[], device: str='cpu', decoding: str='greedy') -> dict:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        assert decoding in ['greedy', 'sampling'], "Decoding must be either 'greedy' or 'sampling'"

        if len(stop) > 0:
            stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop] 
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, device=device)])

        if decoding == 'greedy':
            output = self.model.generate(
                input_ids.to(device), 
                max_new_tokens=512, 
                repetition_penalty = 1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                stopping_criteria=stopping_criteria,
            )
        else:
            output = self.model.generate(    
                input_ids.to(device),
                max_new_tokens=512, 
                do_sample=True,
                top_p = 0.7,
                repetition_penalty = 1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                )

        output = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        output = ' '.join(output.split(prompt)[1:]).strip()

        # parse result
        if not output:
            output = 'None'

        elif output[-1] != ')':
            if output[-1] == ',':
                output = output[:-1] + ')'
            else:
                output = output + ')'
                
        output = {'content': output}
        return output

    @classmethod 
    def set_save_dir(cls, new_save_dir):
        cls.save_dir = new_save_dir


