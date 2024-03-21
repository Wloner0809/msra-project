"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import string
import time
import requests
from openai import OpenAI
import config


def parse_sectioned_prompt(s):
    result = {}
    current_header = None

    for line in s.split("\n"):
        line = line.strip()

        if line.startswith("# "):
            # first word without punctuation(第一个单词去掉标点符号)
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(
                str.maketrans("", "", string.punctuation)
            )
            result[current_header] = ""
        elif current_header is not None:
            result[current_header] += line + "\n"

    return result


def chatgpt(
    prompt,
    temperature=0.7,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=1024,
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
):
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI(
        api_key=config.OPENAI_KEY,
        base_url="https://api.together.xyz/",
    )

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="meta-llama/Llama-2-70b-chat-hf",
        temperature=temperature,
        max_tokens=max_tokens,
        logit_bias=logit_bias,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        stop=stop,
        n=n,
    )
    return [choice.message.content for choice in chat_completion.choices]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
    }
    retries = 0
    while True:
        try:
            r = requests.post(
                "https://api.together.xyz/",
                headers={
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10,
            )
            if r.status_code != 200:
                time.sleep(60)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(60)
            retries += 1
    r = r.json()
    return r["choices"]
