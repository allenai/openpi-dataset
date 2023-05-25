import time
import openai
import numpy as np


def gpt3_inference(prompt: str, model_name: str) -> str:
    while True:
        try:
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
            break
        except openai.error.RateLimitError as e:
            print(e)
            time.sleep(6)

    res = ret["choices"][0]["text"]
    return res


def chat_inference(prompt: str, model_name: str) -> str:
    while True:
        try:
            ret = openai.ChatCompletion.create(
                model=model_name,
                messages=prompt,
                # temperature=0.,
            )
            break
        except openai.error.RateLimitError:
            time.sleep(6)

    res = ret["choices"][0]["message"]
    return res
