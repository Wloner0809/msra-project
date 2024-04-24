import string
from openai import OpenAI
import config
import evaluators
import scorers
import tasks
import argparse


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


def model(
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
        # model="meta-llama/Llama-3-70b-chat-hf",
        # model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        model="Qwen/Qwen1.5-72B-Chat",
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


def get_task_class(task_name):
    if task_name == "ethos":
        return tasks.EthosBinaryTask
    elif task_name == "jailbreak":
        return tasks.JailbreakBinaryTask
    elif task_name == "liar":
        return tasks.DefaultHFBinaryTask
    elif task_name == "ar_sarcasm":
        return tasks.DefaultHFBinaryTask
    else:
        raise Exception(f"Unsupported task: {task_name}")


def get_evaluator(evaluator: str):
    # 论文中的Select()函数
    if evaluator == "bf":
        return evaluators.BruteForceEvaluator
    elif evaluator in {"ucb", "ucb-e"}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {"sr", "s-sr"}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == "sh":
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f"Unsupported evaluator: {evaluator}")


def get_scorer(scorer):
    if scorer == "01":
        return scorers.Cached01Scorer
    else:
        raise Exception(f"Unsupported scorer: {scorer}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, type=bool)
    parser.add_argument("--curriculum", default=False, type=bool)
    parser.add_argument("--task", default="ethos")
    parser.add_argument("--data_dir", default="data/ethos")
    parser.add_argument("--prompts", default="prompts/ethos.md")
    parser.add_argument("--args", default="args/liar_llama3_70B_baseline.json")
    parser.add_argument("--out", default="results/liar_llama3_70B_baseline.json")
    parser.add_argument("--tests", default="tests/liar.json")
    parser.add_argument("--max_threads", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)

    parser.add_argument("--optimizer", default="nl-gradient")
    parser.add_argument("--rounds", default=3, type=int)
    parser.add_argument("--beam_size", default=8, type=int)
    parser.add_argument("--n_test_exs", default=100, type=int)

    parser.add_argument("--minibatch_size", default=16, type=int)
    parser.add_argument("--n_gradients", default=4, type=int)
    parser.add_argument("--errors_per_gradient", default=4, type=int)
    parser.add_argument("--gradients_per_error", default=1, type=int)
    parser.add_argument("--steps_per_gradient", default=1, type=int)
    parser.add_argument("--mc_samples_per_step", default=2, type=int)
    parser.add_argument("--max_expansion_factor", default=8, type=int)

    parser.add_argument("--engine", default="llama", type=str)

    parser.add_argument("--evaluator", default="bf", type=str)
    parser.add_argument("--scorer", default="01", type=str)
    parser.add_argument("--eval_rounds", default=2, type=int)
    parser.add_argument("--eval_prompts_per_round", default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument("--samples_per_eval", default=30, type=int)
    parser.add_argument(
        "--c",
        default=1.0,
        type=float,
        help="exploration param for UCB. higher = more exploration",
    )
    parser.add_argument("--knn_k", default=2, type=int)
    parser.add_argument("--knn_t", default=0.993, type=float)
    parser.add_argument("--reject_on_errors", action="store_true")

    args = parser.parse_args()

    return args
