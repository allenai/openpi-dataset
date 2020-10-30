import operator
import string
from collections import OrderedDict
from typing import Dict, Any

from nltk import PorterStemmer

from src.eval.tw_specific_eval.spacy_stopwords import STOP_WORDS

porter_stemmer = PorterStemmer()


def is_stop(cand:str):
    return cand.lower() in STOP_WORDS or normalize_and_stem(cand) in STOP_WORDS
    # return nlp.vocab[cand].is_stop


def sort_map_by_value(dic: Dict[Any, Any], reverse_order: bool):
    an_ordered_dict = OrderedDict()
    for k, v in sorted(dic.items(), key=operator.itemgetter(1), reverse=reverse_order):
        an_ordered_dict[k] = v
    return an_ordered_dict


def sort_map_by_key(dic: Dict[Any, Any], reverse_order: bool):
    an_ordered_dict = OrderedDict()
    for k, v in sorted(dic.items(), key=operator.itemgetter(0), reverse=reverse_order):
        an_ordered_dict[k] = v
    return an_ordered_dict


def normalize_nostem(s: str) -> str:
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s.strip())))


# Evaluation speed is important.
# Stemming cache for orders of magnitude faster computation
# e.g., stemming of gold phrases was repeated during precision computation
# and stemming of prediction phrases was repeated during recall computation
# The cache is supposed to be very small (at most vocab size map: word-> stem).
# Phrases are generally not queried and even if they were, they wouldn't blow
# up computation at eval time.
cache = dict()
def stem(w: str):
    if not w or len(w.strip()) == 0:
        return ""
    w_lower = w.lower()
    if w_lower in cache:
        return cache[w_lower]
    # Remove leading articles from the phrase (e.g., the rays => rays).
    if w_lower.startswith("a "):
        w_lower = w_lower[2:]
    elif w_lower.startswith("an "):
        w_lower = w_lower[3:]
    elif w_lower.startswith("the "):
        w_lower = w_lower[4:]
    elif w_lower.startswith("your "):
        w_lower = w_lower[5:]
    elif w_lower.startswith("his "):
        w_lower = w_lower[4:]
    elif w_lower.startswith("their "):
        w_lower = w_lower[6:]
    elif w_lower.startswith("my "):
        w_lower = w_lower[3:]
    elif w_lower.startswith("another "):
        w_lower = w_lower[8:]
    elif w_lower.startswith("other "):
        w_lower = w_lower[6:]
    elif w_lower.startswith("this "):
        w_lower = w_lower[5:]
    elif w_lower.startswith("that "):
        w_lower = w_lower[5:]
    # Porter stemmer: rays => ray
    if not w_lower or len(w_lower.strip()) == 0:
        return ""
    ans = porter_stemmer.stem(w_lower).strip()
    cache[w_lower] = ans
    return ans


def normalize_and_stem(s: str) -> str:
        def stemmed(text):
            stemmed_tokens = []
            for token in text.split():
                token_stem = stem(token)
                if token_stem:
                    stemmed_tokens.append(token_stem)
            return ' '.join(stemmed_tokens)

        return stemmed(normalize_nostem(s))



