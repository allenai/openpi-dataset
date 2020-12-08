#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import os

import torch
import sys

from tqdm import tqdm


from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

# probably to avoid "src.xxx" not found?
from training.gen_ans_to_list import aggregate_predictions

sys.path.insert(0, '..')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}


class OpenPIGPT2Predictor:

    def __init__(self, model_path: str, stop_token: str = '<|endoftext|>'):
        self.stop_token = stop_token
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({"sep_token": "[SEP]"})
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        # config = GPT2Config.from_pretrained(args.model_path)
        logger.info(f"Loaded model for generation.")

    def get_predictions_window(self, max_len, k_sentences):
        output_dict = {'window_answers': []}
        next_input = ''
        for input_sent in k_sentences:
            sent = next_input + ' ' + input_sent
            encoded_prompt = self.tokenizer.encode(sent, add_special_tokens=False, return_tensors='pt').to(self.device)
            curr_answer = self.generate_nexttokens_for_sent(max_len=max_len,
                                                            text_so_far=sent,
                                                            encoded_prompt=encoded_prompt)
            next_input += f"{sent} {curr_answer}"
            output_dict['window_answers'].append({"answer": curr_answer.replace('[SEP]', '')})
        return output_dict

    def get_predictions(self, max_len, input_ctxt_and_query):
        encoded_prompt = self.tokenizer.encode(input_ctxt_and_query, add_special_tokens=False, return_tensors='pt')
        encoded_prompt = encoded_prompt.to(self.device)
        answer = self.generate_nexttokens_for_sent(max_len=max_len,
                                                   text_so_far=input_ctxt_and_query,
                                                   encoded_prompt=encoded_prompt)
        return {"answer": answer}

    def generate_nexttokens_for_sent(self, max_len: int, text_so_far: str, encoded_prompt: torch.Tensor) -> str:
        answer: str = ""
        with torch.no_grad():
            out = self.model.generate(
                # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                input_ids=encoded_prompt,
                max_length=max_len + encoded_prompt.size(-1),
                temperature=1.0,
                top_k=0,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
            )

            for out_seq in out:
                text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True)
                text = text[len(text_so_far):]
                text = text[: text.find(self.stop_token) if self.stop_token else None]
                answer += text
                answer += '. '

        return answer


def generate_outfile_path(input_path, output_dir):
    return output_dir.replace("/", "") + "/" + input_path.replace("/", "___")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--max_len",
        default=20,
        type=int,
        help="model path",
    )
    parser.add_argument(
        "--stop_token",
        type=str,
        default='<|endoftext|>',
        help="model path",
    )
    parser.add_argument(
        "--test_input_file",
        default=None,
        type=str,
        help="jsonl file containing id (str) and answer (array) keys",
    )
    parser.add_argument(
        "--test_output_file",
        default=None,
        type=str,
        help="path to store model predictions",
    )
    parser.add_argument(
        "--test_output_agg_file",
        default="",
        required=False,
        type=str,
        help="path to store model predictions aggregated changes per sentence (i.e. turns answers to an array).",
    )

    args = parser.parse_args()

    if not args.model_path or not os.path.exists(args.model_path):
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation model file/dir does not exist: {args.model_path}")
        return

    if not args.test_input_file or not os.path.exists(args.test_input_file):
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation input file does not exist: {args.test_input_file}")
        return

    if not args.test_output_file:
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation output file is empty: {args.test_output_file}")
        return

    args.model_path = args.model_path.strip()
    args.test_output_file = args.test_output_file.strip()
    args.test_input_file = args.test_input_file.strip()

    print(f"Generation task, input = {args.test_input_file}, output = {args.test_output_file} ...")

    predictor = OpenPIGPT2Predictor(model_path=args.model_path, stop_token=args.stop_token)

    test_input = []
    with open(args.test_input_file, 'r') as open_file:
        for line in open_file:
            test_input.append(json.loads(line))

    if os.path.isdir(args.test_output_file):
        args.test_output_file = generate_outfile_path(input_path=args.test_input_file, output_dir=args.test_output_file)

    with open(args.test_output_file, 'w') as open_file:
        for item in tqdm(test_input):
            output = predictor.get_predictions(max_len=args.max_len, input_ctxt_and_query=item['question'])
            output['id'] = item['id']
            json.dump(output, open_file)
            open_file.write('\n')

    aggregated_fp = args.test_output_file + ".aggregated.jsonl" \
        if not args.test_output_agg_file else args.test_output_agg_file
    logger.info(f"Done generating. Aggregating and formatting to {aggregated_fp}")
    aggregate_predictions(prediction_fp=args.test_output_file,
                          out_fp=aggregated_fp)


if __name__ == "__main__":
    main()
