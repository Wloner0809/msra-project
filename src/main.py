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

            # NOTE: 这里暂定的metric为score/perplexity
            # final_metric = [
            #     score / perplexity for score, perplexity in zip(scores, perplexity)
            # ]
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
        prompt = "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # test1
        # llama_liar
        # prompt_baseline = "# Task\nPrioritize the accuracy of the Statement over the speaker's credibility or potential biases. Consider the literal meaning of the Statement, as well as any implicit implications or assumptions. Weigh the evidence provided in the Context, including the speaker's Job title, State, Party, and the circumstances surrounding the Statement. If the Statement is factually incorrect, misleading, or contradicts available evidence, label it as a lie (Yes). Otherwise, label it as truthful (No).\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nWhen assessing the credibility of the Statement, take into account not just its factual correctness, but also the speaker's motivations, attitude, and possible prejudices within the context of the given information. If the Context is unclear or lacks essential details, refrain from making a judgment and instead, ask for further clarification. Only classify the Statement as deceptive (Yes) or genuine (No) when you have sufficient evidence to substantiate your decision.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # mixtral_liar
        # prompt_baseline = "# Task\nExamine the Statement's accuracy in light of the Context, additional information, and the possible motivation or purpose behind it. Decide whether the Statement is misleading (Yes) or genuine (No). For statements that are subjective or intricate, consider the evidence, professional viewpoints, and the prevailing consensus within the relevant field. Also, appraise if the Statement is posed as an actual claim or a potential strategy to influence or mislead the audience.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nConsider the following elements when examining the provided statement: the speaker's job, political allegiance, and current scenario, along with the statement's origin and its possible biases. Note that individuals involved in politics frequently have a personal stake in advancing a specific perspective or plan, which can influence the accuracy of their assertions. Evaluate the reliability of the source and the possible influence of their professional position, political association, and the context of the statement to decide if the statement is deceptive (Yes) or not (No). Bear in mind that editorials, press releases, direct mails, and political advertisements may contain biases and are not consistently trustworthy sources for accurate information.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # qwen_liar
        # prompt_baseline = "# Task\n\"Analyze the claim by thoroughly examining the historical and contextual background, including the specific period mentioned, any public records or official statements made by the individual in the given role, and any corroborating or contradicting data from reliable sources. For political claims, verify not only the Party's stance but also the individual's personal advocacy, voting records, and public speeches at the time in question. When dealing with assertions about policy impacts or numerical forecasts, cross-reference with expert analyses and most recent official data. For personal statements, especially those involving timelines, ensure to align the claim with documented public records of statements to confirm the chronological accuracy. Determine if the Assertion is factually aligned (No) or represents a distortion (Yes), being meticulous about the timeframe and contextual accuracy.\"\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\nYour primary task is to fact-check the core content of each statement. Verify if the claims can be substantiated with existing records or data. Analyze statistical assertions for their accuracy against known datasets. Examine whether the individual's statement aligns with historical events, public records, or official data, regardless of their job title, political stance, or the context in which it was made. Focus on the veracity of the statement itself, though consider any potential misinformation motives only if they directly impact the factuality of the claim. Determine if the statement is factually correct ('No') or intentionally misleading or incorrect ('Yes').\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # test2
        # llama_liar
        # prompt_baseline = "# Task\nBased on the Context and other information, excluding the Job title, State, or Party affiliation, determine whether the Statement is supported by verifiable evidence or facts (No) or lacks credible evidence to back up its claim (Yes).\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = '# Task\nDetermine whether the statement is accurate or inaccurate based on the provided context, including the job title, state, party, and any other relevant information. Label as "Yes" if the statement is inaccurate and "No" if it is accurate.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:'

        # mixtral_liar
        # prompt_baseline = "# Task\nAssess the probability of the Statement being untrue (Yes) or true (No), based on the Provided Context, the Job title, State, and Political affiliation of the individual uttering the Statement. Factual correctness, reliability, and potential intentions of the individual should be taken into account in your evaluation.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        # prompt_curriculum = "# Task\n\"Taking into account the speaker's occupation, geographical context, political leanings, and possible biases or intentions, evaluate whether the Statement contains any distortion, misinterpretation, error, or oversimplification (Yes) or not (No). Thoroughly analyze the statement's implications and intricacies, and judge if it precisely portrays the circumstances or subject matter it aims to address.\"\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

        # qwen_liar
        prompt_baseline = "# Task\nGiven an assertion, assess its credibility by categorizing it as accurate (No) if it is supported by factual evidence, historical facts, or verifiable records, or as deceptive (Yes) if it includes false information, baseless speculations, or misrepresents potential outcomes. Evaluate if the claim can be substantiated with existing data or pertains to a verifiable commitment, pledge, or occurrence. Ignore the authorship of the statement and center your judgment on the information's reliability, also considering the合理性 of predicted scenarios according to present understanding.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"
        prompt_curriculum = "# Task\nScrutinize the statement to discern its nature as a factual claim or a subjective perspective, taking into account the context of policy intentions, future projections, and the source's potential biases. Verify statistical assertions against reliable data sources, and assess policy statements based on declared objectives, feasibility studies, and existing frameworks rather than assuming an outcome. For statements involving estimates, intentions, or projections, consider if they are presented as fact or as a goal. When evaluating statements by public figures, discern if they are describing actions, plans, or outcomes that are verifiable now or are conditional on future events. Conclude by categorizing the assertion as fact (No) if it aligns with verifiable evidence, or as a misstatement (Yes) if it is misleading, speculative, or contrary to established knowledge.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {{ text }}\nLabel:"

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
