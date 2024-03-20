"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import string
import time
import requests

import config


def parse_sectioned_prompt(s):
    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation(第一个单词去掉标点符号)
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


# TODO: 这里调用openai api代码可以参考OPRO
def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024,
            presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }
    retries = 0  # 感觉没用到
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions',
                              headers={
                                  "Authorization": f"Bearer {config.OPENAI_KEY}",
                                  "Content-Type": "application/json"
                              },
                              json=payload,
                              timeout=timeout
                              )
            if r.status_code != 200:
                print(r.status_code, r.text)
                retries += 1
                time.sleep(60)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(60)
            retries += 1
    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    retries = 0
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                              headers={
                                  "Authorization": f"Bearer {config.OPENAI_KEY}",
                                  "Content-Type": "application/json"
                              },
                              json=payload,
                              timeout=10
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
    return r['choices']
