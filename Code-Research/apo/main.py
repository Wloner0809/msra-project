import argparse
import json
import os
import time

from tqdm import tqdm

import evaluators
import optimizers
import predictors
import scorers
import tasks


def get_task_class(task_name):
    if task_name == 'ethos':
        return tasks.EthosBinaryTask
    elif task_name == 'jailbreak':
        return tasks.JailbreakBinaryTask
    elif task_name == 'liar':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'ar_sarcasm':
        return tasks.DefaultHFBinaryTask
    else:
        raise Exception(f'Unsupported task: {task_name}')


def get_evaluator(evaluator: str):
    # 论文中的Select()函数
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')


def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'll':
        return scorers.CachedLogLikelihoodScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='ethos')
    parser.add_argument('--data_dir', default='data/ethos')
    parser.add_argument('--prompts', default='prompts/ethos.md')
    parser.add_argument('--out', default='ethos_results_70B.json')
    parser.add_argument('--max_threads', default=1, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=3, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=400, type=int)

    parser.add_argument('--minibatch_size', default=16, type=int)
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=4, type=int)
    parser.add_argument('--eval_prompts_per_round', default=4, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    config = vars(args)  # 转为字典

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']

    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(',')]

    json_dict = {}
    json_dict['round'] = []
    json_dict['time'] = []
    json_dict['candidates'] = []
    json_dict['estimated_scores'] = []
    json_dict['metrics'] = []

    for round in tqdm(range(config['rounds'] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)

        # score candidates
        scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]

        # evaluate candidates
        metrics = []
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
            metrics.append(f1)

        # save results
        json_dict['round'].append(round)
        json_dict['time'].append(time.time() - start)
        json_dict['candidates'].append(candidates)
        json_dict['estimated_scores'].append(scores)
        json_dict['metrics'].append(metrics)
        with open(args.out, 'w', encoding='utf-8') as outf:
            json.dump(json_dict, outf, ensure_ascii=False, indent=4)

    print("DONE!")
