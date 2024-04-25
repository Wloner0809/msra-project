import json
import os
import time
from tqdm import tqdm

import optimizers
import predictors
from utils import get_args, get_evaluator, get_scorer, get_task_class


if __name__ == "__main__":
    args = get_args()

    config = vars(args)  # 转为字典
    config["eval_budget"] = (
        config["samples_per_eval"]
        * config["eval_rounds"]
        * config["eval_prompts_per_round"]
    )

    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator("bf")(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.ProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    if args.train:
        if os.path.exists(args.out):
            os.remove(args.out)
        if os.path.exists(args.args):
            os.remove(args.args)

        print(config)
        with open(args.args, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        candidates = [open(fp.strip()).read() for fp in args.prompts.split(",")]

        json_dict = {}
        json_dict["round"] = []
        json_dict["time"] = []
        json_dict["candidates"] = []
        json_dict["estimated_scores"] = []
        json_dict["f1"] = []
        json_dict["acc"] = []

        for round in tqdm(range(config["rounds"] + 1)):
            print("STARTING ROUND ", round)
            start = time.time()

            # expand candidates
            if round > 0:
                candidates = optimizer.expand_candidates(
                    candidates,
                    task,
                    gpt4,
                    train_exs,
                    seed=round,
                    curriculum=args.curriculum,
                )

            # score candidates
            scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
            [scores, candidates] = list(
                zip(*sorted(list(zip(scores, candidates)), reverse=True))
            )

            # select candidates
            candidates = candidates[: config["beam_size"]]
            scores = scores[: config["beam_size"]]

            # evaluate candidates
            F1 = []
            Acc = []
            for candidate, score in zip(candidates, scores):
                acc, f1, texts, labels, preds = task.evaluate(
                    gpt4, candidate, test_exs, n=args.n_test_exs
                )
                F1.append(f1)
                Acc.append(acc)

            # save results
            json_dict["round"].append(round)
            json_dict["time"].append(time.time() - start)
            json_dict["candidates"].append(candidates)
            json_dict["estimated_scores"].append(scores)
            json_dict["f1"].append(F1)
            json_dict["acc"].append(Acc)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(json_dict, f, ensure_ascii=False, indent=4)

        print("FINISHED")
    else:
        # llama_liar
        # prompt_baseline = '# Task\nHere is a variation of the instruction:\n\nInput: Analyze the Statement in the context of the provided Context, Job title, State, and Party affiliation, taking into account both the statement\'s factual correctness and its potential impact, attitude, and delivery. Assess whether the Statement provides a neutral, accurate, and reliable representation of the facts, or if it manipulates, misrepresents, or distorts information to deceive, mislead, or create a misleading narrative. If the Statement makes a factual claim that is verifiably true, unbiased, and free of misleading implications, categorize it as "No". If the Statement makes a misleading or false claim that presents inaccurate, exaggerated, or distorted information, implies a false narrative, omits critical context, or uses emotional manipulation, categorize it as "Yes".\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:'
        # prompt_curriculum = "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) by carefully analyzing the Context, Job title, State, and Party affiliation, and considering the medium of communication (e.g. email, TV ad, campaign speech, robocall). When evaluating the Statement, consider not only factual inaccuracies but also the potential for misleading implications, exaggerations, or omissions that could alter the interpretation of the information. Pay particular attention to how the Statement might be perceived by the intended audience, and whether it presents a distorted or incomplete picture of reality. If the Statement contains both accurate and inaccurate information, classify it as a lie (Yes) if the inaccurate or misleading part has a significant impact on the overall meaning or implication of the Statement.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # mixtral_liar
        # prompt_baseline = "# Task\n  Assess the truthfulness of the Claim (Respond with Yes if it's a falsehood, and No if it's accurate) by utilizing the given Information, supplementary details, and considering if the claim is misleading due to the absence of vital information, other deceitful strategies, or the speaker's inherent bias/viewpoint.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\n \"Assess the truthfulness of the Claim, taking into account the provided Context, Occupation, State, Political Affiliation, and the potential bias of the Context. Consider logical consistency, factual accuracy, contextual relevance, and reliable sources to make your determination. Respond with 'Yes' if the claim is false and 'No' if the claim is true.\"\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # qwen_liar
        # prompt_baseline = "# Task\nEvaluate the truthfulness of the provided Statement, considering the information in the context, the reliability of the source, and any available data. If the statement can be conclusively proven or disproven based on the content alone, regardless of the job title, state, or party mentioned, mark it as either false (Yes) or truthful (No).\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nAssess the truthfulness of the claim by examining the Context, supporting data, and the credibility of the source's knowledge or authority, while remaining impartial to their profession, geographic origin, or ideological alignment. Weigh the significance of factual coherence alongside the understanding that some sources may possess specialized insights or stances that affect their reliability. Conclude whether the statement is likely to be authentic (No) or possibly a misrepresentation or untruth (Yes).\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # llama_ethos
        # prompt_baseline = "# Task\nOutput: Classify the text as hate speech if it expresses discriminatory or violent intentions towards an individual or group based on their race, ethnicity, nationality, religion, gender identity, sexual orientation, disability, or any other legally protected characteristic. Consider the context and intent behind the language, rather than just the presence of certain keywords. Look for language that promotes stereotypes, stigma, or harm towards a protected group, and avoid misclassifying texts that use protected characteristics to criticize discriminatory attitudes or promote inclusivity.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nHere's a variation of the instruction:\n\nInput: Output: Does the given text exhibit any language or tone that promotes bias, prejudice, or hostility towards a person or group due to their intrinsic traits, such as race, ethnicity, gender, religious beliefs, sexual identity, or other personal characteristics, whether the language is overt or subtle?\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # qwen_ethos
        prompt_baseline = "# Task\nIdentify if the text contains language that explicitly promotes hatred, discrimination, or advocates for violence against an individual or group due to their race, ethnicity, nationality, religion, sex, gender identity, sexual orientation, or any other protected characteristic. Analyze the context to ensure the criticism is not aimed at a specific ideology, organization, or action rather than the attributes of the protected group itself.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        prompt_curriculum = "# Task\nEvaluate the text for the presence of language that expresses disdain, prejudice, or hostility towards an individual, community, or a safeguarded attribute such as race, religion, or gender, while taking into account the context and the intent behind the message. Distinguish between genuine expressions of hate speech and forms of irony, sarcasm, parody, or critical commentary, especially when discussing organizations or movements. Avoid misinterpreting constructive criticism or opinion-based statements as hate speech.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        acc, f1, texts, labels, preds = task.evaluate(
            gpt4, prompt_baseline, test_exs, len(test_exs)
        )
        print("-" * 30 + "Baseline" + "-" * 30)
        print(f"F1: {f1}")
        print(f"acc:{acc}")
        json_dict = {}
        json_dict["baseline"] = {}
        json_dict["curriculum"] = {}
        json_dict["baseline"]["F1"] = f1
        json_dict["baseline"]["Acc"] = acc

        acc, f1, texts, labels, preds = task.evaluate(
            gpt4, prompt_curriculum, test_exs, len(test_exs)
        )
        print("-" * 30 + "Curriculum" + "-" * 30)
        print(f"F1: {f1}")
        print(f"acc:{acc}")
        json_dict["curriculum"]["F1"] = f1
        json_dict["curriculum"]["Acc"] = acc
        with open(args.tests, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
