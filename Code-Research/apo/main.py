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
        # prompt_baseline = "# Task\nHere's a revised instruction:\n\nInput: Evaluate the credibility of the Statement by thoroughly examining the Context, Job title, State, Party, and supplementary facts, alongside the Statement itself. Consider not only intentional falsehoods but also statements that are misleading, lack crucial context, or make unsubstantiated claims. Use all provided information to make an informed judgment about the Statement's veracity.\n\nOutput: Classify the Statement as either truthful (No) or deceptive (Yes), where deceptive includes statements that are intentionally false, misleading, lack context, or make unsubstantiated claims.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nHere's a variation of the instruction:\n\n**Variation 2**\nWhen assessing the Statement, weigh the significance of the speaker's Job title, their location in a particular State, their political Party affiliation, and the Context surrounding the statement. Delve into the subtleties of the situation and the potential consequences of the statement. If the Statement lacks concrete evidence or is disproven by verifiable facts, classify it as inaccurate (Yes). Otherwise, classify it as accurate (No).\n\nLet me know if you'd like me to generate more variations!\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # mixtral_liar
        prompt_baseline = "# Task\n Ignoring the Position title, Location, and Affiliation, assess the Statement and its Context to determine whether the Statement constitutes a likely political hyperbole or untrue assertion (Yes) or not (No), taking into account the nature of political campaigns and addresses.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        prompt_curriculum = "# Task\n Leveraging the Context, Job Role, State, and Party details provided, assess the reliability and truthfulness of the Statement. If the statement aligns with the provided information, categorize it as not a lie (No). However, if it contradicts or lacks support from the given data, classify it as a lie (Yes).\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

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
