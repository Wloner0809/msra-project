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

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(",")]

    json_dict = {}
    json_dict["round"] = []
    json_dict["time"] = []
    json_dict["candidates"] = []
    json_dict["estimated_scores"] = []
    json_dict["metrics"] = []

    for round in tqdm(range(config["rounds"] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)

        # score candidates
        scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
        [scores, candidates] = list(
            zip(*sorted(list(zip(scores, candidates)), reverse=True))
        )

        # select candidates
        candidates = candidates[: config["beam_size"]]
        scores = scores[: config["beam_size"]]

        # evaluate candidates
        metrics = []
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds = task.evaluate(
                gpt4, candidate, test_exs, n=args.n_test_exs
            )
            metrics.append(f1)

        # save results
        json_dict["round"].append(round)
        json_dict["time"].append(time.time() - start)
        json_dict["candidates"].append(candidates)
        json_dict["estimated_scores"].append(scores)
        json_dict["metrics"].append(metrics)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)

    print("DONE!")
