import operator
import string
from collections import OrderedDict
from typing import Dict, Any


from eval.spacy_stopwords import STOP_WORDS

from eval.porter import PorterStemmer

porter_stemmer = PorterStemmer()

def is_stop(cand: str):
    # return nlp.vocab[cand].is_stop
    return cand.lower() in STOP_WORDS or normalize_and_stem(cand) in STOP_WORDS


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
leads = ["a ", "an ", "the ", "your ", "his ", "their ", "my ", "another ", "other ", "this ", "that "]


def stem(w: str):
    if not w or len(w.strip()) == 0:
        return ""
    w_lower = w.lower()
    if w_lower in cache:
        return cache[w_lower]
    # Remove leading articles from the phrase (e.g., the rays => rays).
    for lead in leads:
        if w_lower.startswith(lead):
            w_lower = w_lower[len(lead):]
            break
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
