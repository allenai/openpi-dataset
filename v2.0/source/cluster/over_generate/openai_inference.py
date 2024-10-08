import openai
import backoff
from typing import Dict, List


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gpt3_inference(prompt: str, model_name: str) -> str:
    ret = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=0.,
        max_tokens=200,
        top_p=1,
        logprobs=5,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n"]
    )
    res = ret["choices"][0]["text"]
    return res


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_inference(prompt: list[Dict[str, str]]) -> Dict[str, str]:
    ret = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompt,
        temperature=0.7,
        max_tokens=256,
        stop=["\n\n"]
    )
    res = ret["choices"][0]["message"]
    return res
