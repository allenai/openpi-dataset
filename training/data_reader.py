import json
import logging
import os
import pickle
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TWSentBySentDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512, skip_answer=False, cache_dir=None):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        if cache_dir is None:
            cache_dir = directory
        cached_features_file = os.path.join(cache_dir, f'cached_input_feat_{block_size}_{filename}')
        cached_metadata_file = os.path.join(cache_dir, f'cached_input_meta_{block_size}_{filename}')

        if os.path.exists(cached_features_file) and os.path.exists(cached_metadata_file):
            logger.info(f"Loading features and metadata from cache: {cached_features_file} and {cached_metadata_file}")
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            with open(cached_metadata_file, 'rb') as handle:
                self.metadata = pickle.load(handle)
        else:
            logger.info(f"Creating features from dataset file at {file_path} and directory {directory}")
            self.examples = []
            self.metadata = []
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    input_json = json.loads(line)
                    token_ids, token_labels, one_metadata = TWSentBySentDataset.read_line(input_json=input_json,
                                                                                          tokenizer=tokenizer,
                                                                                          block_size=block_size,
                                                                                          skip_answer=skip_answer)

                    self.examples.append((token_ids, token_labels, one_metadata))
                    self.metadata.append(one_metadata)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cached_metadata_file, 'wb') as handle:
                pickle.dump(self.metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _truncate_seq_pair(tokens_q, tokens_ans, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # In all versions of TrackWorld datasets (even entity+attr),
        # the answer part is really short (and important)
        # x was --- before and --- afterwards.
        # while the context keeps growing (more and more previous sentences become context,
        # context reaches upto about 9 sentences for the last step).
        # so, it is okay to lose the question context a bit but not the answer.
        while True:
            total_length = len(tokens_q) + len(tokens_ans)
            if total_length <= max_length:
                break
            elif len(tokens_q) > 0:
                # initial sentences in the context are less important than latter sentences.
                tokens_q.pop(0)
            elif len(tokens_ans) > 0:
                tokens_ans.pop(0)
            else:
                raise Exception(f"token_ans are empty, and nothing more to truncate from question.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item_idx):
        return (torch.tensor(self.examples[item_idx][0]),
                torch.tensor(self.examples[item_idx][1]),
                torch.tensor([item_idx]))

    def get_original_id_for(self, item_idx):
        return self.metadata[item_idx]['id']

    @staticmethod
    def read_line(input_json, tokenizer, block_size, skip_answer, stop_token='<|endoftext|>'):
        metadata = {}
        tokenized_question = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_json['question']))
        if not skip_answer:
            if 'answer' not in input_json or input_json['answer'] == '':
                tokenized_answer = []
            else:
                tokenized_answer = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(input_json['answer'])) if not skip_answer else []
        else:
            tokenized_answer = []

        metadata['id'] = input_json['id']
        TWSentBySentDataset._truncate_seq_pair(tokenized_question, tokenized_answer, max_length=block_size - 1)

        if not skip_answer:
            token_ids = tokenized_question + tokenized_answer + tokenizer.convert_tokens_to_ids([tokenizer.eos_token])
        else:
            token_ids = tokenized_question

        token_labels = [-100] * len(tokenized_question) + token_ids[len(tokenized_question):]

        if len(token_ids) < block_size:
            add_tokens = block_size - len(token_ids)
            token_ids = token_ids + [0] * add_tokens
            token_labels = token_labels + [-100] * add_tokens

        if len(token_ids) > block_size:  # Truncate in block of block_size
            raise ValueError("Unexpected #tokens ({}) > block size ({}).".format(
                len(token_ids), block_size))

        assert len(token_ids) == len(token_labels)

        return token_ids, token_labels, metadata


def load_and_cache_examples(tokenizer, output_dir: str, file_path: str, block_size: int):
    return TWSentBySentDataset(tokenizer=tokenizer,
                               file_path=file_path,
                               block_size=block_size,
                               cache_dir=output_dir)
