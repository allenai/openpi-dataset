import time
import openai
import backoff
import numpy as np


# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gpt3_inference(prompt: str, model_name: str) -> str:
    while True:
        try:
            ret = openai.Completion.create(
                model=model_name,
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
                top_p=1,
                logprobs=5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n\n"]
            )
            break
        except openai.error.RateLimitError as e:
            print(e)
            time.sleep(5)

    res = ret["choices"][0]["text"]
    return res


# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_inference(prompt: str, model_name: str) -> str:
    while True:
        try:
            ret = openai.ChatCompletion.create(
                model=model_name,
                messages=prompt
            )
            break
        except openai.error.RateLimitError as e:
            print(e)
            time.sleep(5)

    res = ret["choices"][0]["message"]
    return res
