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

from training.gen_ans_to_list import aggregate_predictions

# to avoid "src.xxx" not found error.
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
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Fixed GPT2 tokenizer.
        self.tokenizer.add_special_tokens({"sep_token": "[SEP]"})
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model for generation.")

    def get_predictions(self, max_len, input_ctxt_and_query, temperature: float = 1.0,
                        top_k: int = 0,
                        top_p: float = 0.9,
                        do_sample: bool = True,
                        num_return_sequences: int = 1):
        '''
        :param max_len: max number of tokens to generate overall.
        :param  top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.
                Must be between 0 and 1. Default to 0.9
        :param  temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        :param  top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 0 and infinity. Defaults to 0
        :param  num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.
        :param do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False`.
        :return: generated next token.
        '''
        encoded_prompt = self.tokenizer.encode(input_ctxt_and_query, add_special_tokens=False, return_tensors='pt')
        encoded_prompt = encoded_prompt.to(self.device)
        answer = self.generate_nexttokens_for_sent(max_len=max_len,
                                                   text_so_far=input_ctxt_and_query,
                                                   encoded_prompt=encoded_prompt,
                                                   temperature=temperature,
                                                   top_k=top_k,
                                                   top_p=top_p,
                                                   do_sample=do_sample,
                                                   num_return_sequences=num_return_sequences
                                                   )
        return {"answer": answer}

    def generate_nexttokens_for_sent(self,
                                     max_len: int,
                                     text_so_far: str,
                                     encoded_prompt: torch.Tensor,
                                     temperature: float,
                                     top_k: int,
                                     top_p: float,
                                     do_sample: bool,
                                     num_return_sequences) -> str:
        '''
        :param text_so_far: text generated so far.
        :param encoded_prompt: `tf.Tensor` of `dtype=tf.int32` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `tf.Tensor` of shape `(1,)`.
        :return: generated next token.
        '''
        answer: str = ""
        with torch.no_grad():
            out = self.model.generate(
                # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                input_ids=encoded_prompt,
                max_length=max_len + encoded_prompt.size(-1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences
            )

            for out_seq in out:
                text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True)
                text = text[len(text_so_far):]
                text = text[: text.find(self.stop_token) if self.stop_token else None]
                answer += text
                answer += '. '

        return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="model path",
        required=True
    )
    parser.add_argument(
        "--max_len",
        default=20,
        type=int,
        help="model path",
        required=True
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
        help="jsonl file containing id (str) and question (str) keys",
        required=True
    )
    parser.add_argument(
        "--unformatted_outpath",
        default=None,
        type=str,
        help="path to store unformatted model predictions",
        required=True
    )
    parser.add_argument(
        "--formatted_outpath",
        default="",
        type=str,
        help="path to store formatted model predictions (i.e. turns a string answer to an array of state changes).",
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

    if not args.unformatted_outpath:
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation output file is empty: {args.unformatted_outpath}")
        return

    args.model_path = args.model_path.strip()
    args.unformatted_outpath = args.unformatted_outpath.strip()
    args.test_input_file = args.test_input_file.strip()

    print(f"Generation task, input = {args.test_input_file}, output = {args.unformatted_outpath} ...")

    predictor = OpenPIGPT2Predictor(model_path=args.model_path, stop_token=args.stop_token)

    test_input = []
    with open(args.test_input_file, 'r') as open_file:
        for line in open_file:
            test_input.append(json.loads(line))

    with open(args.unformatted_outpath, 'w') as open_file:
        for item in tqdm(test_input):
            output = predictor.get_predictions(max_len=args.max_len, input_ctxt_and_query=item['question'])
            output['id'] = item['id']
            json.dump(output, open_file)
            open_file.write('\n')

    formatted_fp = args.unformatted_outpath + ".formatted.jsonl" \
        if not args.formatted_outpath else args.formatted_outpath
    logger.info(f"Done generating. Aggregating and formatting to {formatted_fp}")
    aggregate_predictions(prediction_fp=args.unformatted_outpath,
                          out_fp=formatted_fp)


if __name__ == "__main__":
    main()
