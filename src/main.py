import json
import os
import time
import numpy as np
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
        # NOTE: 初始perplexity默认为1
        candidates = [(candidate, 1) for candidate in candidates]

        json_dict = {}
        json_dict["round"] = []
        json_dict["time"] = []
        json_dict["candidates"] = []
        json_dict["estimated_scores"] = []
        json_dict["metric"] = []

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
            candidate_prompt = [candidate[0] for candidate in candidates]
            scores = optimizer.score_candidates(candidate_prompt, task, gpt4, train_exs)
            perplexity = [candidate[1] for candidate in candidates]

            final_metric = scores
            metric_index = np.argsort(final_metric)[::-1]
            candidate_new_metric = []
            score_new_metric = []
            metric = []
            for i in range(min(config["beam_size"], len(metric_index))):
                candidate_new_metric.append(candidates[metric_index[i]])
                score_new_metric.append(scores[metric_index[i]])
                metric.append(scores[metric_index[i]] / perplexity[metric_index[i]])

            # [scores, candidates] = list(
            #     zip(*sorted(list(zip(scores, candidates)), reverse=True))
            # )

            # select candidates
            # candidates = candidates[: config["beam_size"]]
            # scores = scores[: config["beam_size"]]
            candidates = candidate_new_metric
            scores = score_new_metric

            # save results
            json_dict["round"].append(round)
            json_dict["time"].append(time.time() - start)
            json_dict["candidates"].append(candidates)
            json_dict["estimated_scores"].append(scores)
            json_dict["metric"].append(metric)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(json_dict, f, ensure_ascii=False, indent=4)

        print("FINISHED")
    else:
        "liar dataset"
        # llama
        # prompt_baseline = "# Task\nDetermine whether the given Statement is trustworthy or not. Label it as 'Deceptive' (Yes) if it intentionally provides false, inaccurate, or misleading information, or 'Accurate' (No) if it is a genuine and reliable statement. When making your judgment, take into account the provided Context, Job title, State, and Party of the speaker, as well as any external information that may be relevant to the claim.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nIgnore the Job title, State, and Party affiliation, and focus solely on the Context and Statement. Classify the Statement as either a factual inaccuracy or misleading information (Yes) or a true and accurate claim (No).\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # mixtral
        # prompt_baseline = "# Task\nConsider the Statement, Job title, State, Party, and Context. Determine whether the Statement is a lie (Yes) or not (No), taking into account the potential biases or motivations of the speaker. Keep in mind that politicians or public figures may make statements that are technically true but misleading or taken out of context.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nEvaluate the Statement in light of the Context, Job title, State, Party, and any other relevant information. Ascertain whether the Statement is a verifiable fact (No) or a subjective viewpoint (Yes) that could be swayed by the speaker's political leanings or ideology, thereby potentially incorporating bias or partisanship.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # qwen
        # prompt_baseline = "# Task\nExamine the credibility of the assertion by considering the historical backdrop, shifts in regulations or personal viewpoints through time, the individual's professional position, their alignment with a political State and Party, notable events referenced, and all relevant information. Judge whether the assertion is misleading (Yes) or appears valid (No), taking into account chronological changes and the development of situations.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nEvaluate the truthfulness of the Statement by considering the Context, the Job title of the speaker, their Party affiliation, and any historical or widely known facts. Determine if the Statement is a misrepresentation or aligns with reality, and label it as a lie (Yes) if it is factually incorrect or misleading based on these factors, and not a lie (No) otherwise.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        "sst2 dataset"
        # llama
        prompt_baseline = "# Task\nConsider the nuances of language and the context in which words are used to determine the overall sentiment of the given text. Look for phrases or sentences that may convey irony, sarcasm, or contrast to accurately identify whether the sentiment leans towards being positive or negative. Respond with 'Yes' for positive and 'No' for negative.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        prompt_curriculum = "# Task\nDetermine whether the given text expresses a positive or affirmative stance (Yes) or a negative or disapproving stance (No) towards a specific topic or idea.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # mixtral
        # prompt_baseline = '# Task\n"Analyze the sentiment of the following statement, considering both positive and negative aspects, and determine if the overall sentiment is favorable (Yes) or unfavorable (No)."\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:'
        # prompt_curriculum = '# Task\n"Evaluate the sentiment conveyed in the subsequent statement, taking into account both overt and subtle hints like sarcasm or irony. Decide whether the sentiment is positive (Yes) or negative (No)."\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:'

        # qwen
        # prompt_baseline = '# Task\nAssess the prevailing emotional tone of the passage as favorable (Yes) or unfavorable (No), being mindful of any coexisting positive and negative sentiments or situational contradictions. Consider phrases that signal sentiment reversals, such as "on the other hand" or "despite that", to render an accurate evaluation that incorporates the text\'s nuanced emotional balance.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:'
        # prompt_curriculum = "# Task\nYour task is to analyze the sentiment within a given text by considering the full context and subtle implications of the words used. Decide if the opinion conveyed is positive (respond with 'Yes') or negative (respond with 'No'). Be meticulous about identifying any words or expressions that might introduce a contrasting sentiment. ## Example 1\\nText: \"an otherwise delightfully comic narrative\"\\nLabel: Yes\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

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
