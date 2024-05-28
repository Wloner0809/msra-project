import random
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import utils


class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs, seed):
        pass


class ProTeGi(PromptOptimizer):
    """ProTeGi: Prompt Optimization with Textual Gradients"""

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _curriculum_learning(self, minibatch_size, minibatch, n):
        minibatch_str = ""
        for i in range(minibatch_size):
            minibatch_str += f"{minibatch[i]['text']} \n"

        curriculum_prompt = f"""
        I'm trying to find out the difficulty level of the following {minibatch_size} problems.

        Determine whether the following statement is a lie or not:
        "{minibatch_str}"

        You must output all numbers between 1 and {minibatch_size} exactly once according to difficulty level, where 1 means easiest and {minibatch_size} means most difficult.
        
        Wrap your response with <START> and <END>, you should separate the ranks with commas.
        """

        curriculum_prompt = "\n".join(
            [line.lstrip() for line in curriculum_prompt.split("\n")]
        )
        responses = utils.teacher(curriculum_prompt, n=n)
        rank = []
        for response in responses:
            rank += self.parse_tagged_text(response, "<START>", "<END>")
        return rank

    def _rank_error(self, error_num, error_str, n):
        curriculum_prompt = f"""
        I'm trying to find out the difficulty level of the following {error_num} problems.

        Determine whether the following statement is a lie or not:
        "{error_str}"

        You must only output all numbers between 1 and {error_num} exactly once according to difficulty level, where 1 means easiest and {error_num} means most difficult.
        
        Wrap your ranks with <START> and <END>, you should separate the ranks with commas.
        """
        curriculum_prompt = "\n".join(
            [line.lstrip() for line in curriculum_prompt.split("\n")]
        )
        responses = utils.teacher(curriculum_prompt, n=n)
        rank = []
        for response in responses:
            rank += self.parse_tagged_text(response, "<START>", "<END>")
        return rank

    def _sample_error_str(
        self, texts, labels, preds, task, seed, n=4, curriculum=False
    ):
        """Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (label, pred) in enumerate(zip(labels, preds)):
            if label != pred:
                error_idxs.append(i)

        self._set_seed(seed)
        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))
        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]

        # NOTE: Curriculum learning is used here
        sample_str = ""
        for i in range(len(sample_texts)):
            sample_str += f"{sample_texts[i]} \n"
        if curriculum:
            rank = self._rank_error(len(sample_texts), sample_str, 1)
            while rank == []:
                rank = self._rank_error(len(sample_texts), sample_str, 1)
            if rank != []:
                index = np.argsort(rank[0].split(", "))
                if len(index) > len(sample_texts):
                    index = [i for i in index if i < len(sample_texts)]
                sample_texts = [sample_texts[i] for i in index]
                sample_labels = [sample_labels[i] for i in index]
                sample_preds = [sample_preds[i] for i in index]
                # # NOTE: 打印error的顺序
                # print("error order: ", index)
                # print("rank: ", rank[0].split(", "))

        error_string = ""
        error_idx = 0
        for i, (text, label, pred) in enumerate(
            zip(sample_texts, sample_labels, sample_preds)
        ):
            error_string += f"## Example {error_idx + 1}\n"
            error_string += f'Text: "{text.strip()}"\nLabel: {task.stringify_prediction(label)}\nPrediction: {task.stringify_prediction(pred)}\n\n'
            error_idx += 1
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index + len(end_tag) :]
        return texts

    def _get_gradients(
        self, prompt, error_string, num_feedbacks=5, n=1, curriculum=False
    ):
        """Get "gradients" for a prompt based on the error string."""
        if curriculum:
            gradient_prompt = f"""
            I'm trying to write a zero-shot classifier prompt.
        
            My current prompt is:
            "{prompt}"

            But this prompt gets the following examples wrong, these examples are ranked by difficulty level from easy to hard:
            {error_string}

            Give {num_feedbacks} different reasons why the prompt could have gotten these examples wrong.
            Wrap each reason with <START> and <END>
            """
        else:
            gradient_prompt = f"""
            I'm trying to write a zero-shot classifier prompt.
        
            My current prompt is:
            "{prompt}"

            But this prompt gets the following examples wrong:
            {error_string}

            Give {num_feedbacks} different reasons why the prompt could have gotten these examples wrong.
            Wrap each reason with <START> and <END>
            """
        gradient_prompt = "\n".join(
            [line.lstrip() for line in gradient_prompt.split("\n")]
        )  # 去除开头的空格
        responses, log_prob = utils.model(gradient_prompt, n=n)
        feedbacks = []
        for response in responses:
            feedbacks += self.parse_tagged_text(response, "<START>", "<END>")
        # for prob in log_prob:
        #     print("Feedback perplexity: ", utils.calculate_perplexity(prob))
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        """Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.
        
        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Wrap each prompt with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = "\n".join(
            [line.lstrip() for line in transformation_prompt.split("\n")]
        )
        responses, log_prob = utils.model(transformation_prompt, n=n)
        new_prompts = []
        perplexity = []
        for prob in log_prob:
            perplexity.append(utils.calculate_perplexity(prob))
            # print("Prompt perplexity: ", utils.calculate_perplexity(prob))
        for i, response in enumerate(responses):
            new_prompts.append(
                (
                    self.parse_tagged_text(response, "<START>", "<END>"),
                    perplexity[i],
                )
            )
        return new_prompts

    # 产生语义相似的句子, 用于增加prompt数目
    def generate_synonyms(self, prompt_section, n=3):
        """Generate synonyms for a prompt section."""
        rewrite_prompt = f"""
        I'm trying to write a variation of the following instruction while keeping the semantic meaning.
        
        The instruction is:
        {prompt_section}
        
        Wrap each prompt with <START> and <END>.
        
        The variation of the instruction is:
        """
        new_instructions, log_prob = utils.model(rewrite_prompt, n=n)
        instruction = []
        perplexity = []
        for prob in log_prob:
            perplexity.append(utils.calculate_perplexity(prob))
            # print("Semantic prompt perplexity: ", utils.calculate_perplexity(prob))
        for i, new_instruction in enumerate(new_instructions):
            instruction.append(
                (
                    self.parse_tagged_text(new_instruction, "<START>", "<END>"),
                    perplexity[i],
                )
            )
        return instruction

    def get_gradients(
        self, prompt, task_section, task, gpt4, texts, labels, preds, seed, curriculum
    ):
        """Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="gradients..",
        ):
            error_string = self._sample_error_str(
                texts,
                labels,
                preds,
                task,
                seed,
                n=self.opt["errors_per_gradient"],
                curriculum=curriculum,
            )
            gradients = self._get_gradients(
                task_section,
                error_string,
                self.opt["gradients_per_error"],
                n=1,
                curriculum=curriculum,
            )
            prompt_feedbacks += [(gradient, error_string) for gradient in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, gpt4, train_exs, seed, curriculum=False):
        """Expand a list of prompts by generating gradient-based successors and
        synonyms for each section.
        """
        # NOTE: Curriculum learning is used here
        self._set_seed(seed)
        minibatch = random.sample(train_exs, k=self.opt["minibatch_size"])

        if curriculum:
            rank = self._curriculum_learning(self.opt["minibatch_size"], minibatch, n=1)
            while rank == []:
                rank = self._curriculum_learning(
                    self.opt["minibatch_size"], minibatch, n=1
                )
            if rank != []:
                index = np.argsort(rank[0].split(","))
                # NOTE: 打印出来minibatch的顺序
                # print("minibatch order: ", index)
                if len(index) > len(minibatch):
                    minibatch = [minibatch[i] for i in index if i < len(minibatch)]
                minibatch = [minibatch[i] for i in index]

        new_prompts = []
        for prompt in tqdm(prompts, desc=f"expanding {len(prompts)} prompts"):
            sections = utils.parse_sectioned_prompt(prompt[0])
            task_section = sections["task"].strip()

            # evaluate prompt on minibatch
            _, _, texts, labels, preds = task.evaluate(gpt4, prompt[0], minibatch)

            """
            new_task_sections是通过gradients得到的新的task_sections
            mc_sampled_task_sections是通过generate_synonyms得到的新的task_sections
            分别对应论文Expand()算法中的4、5步
            """
            # get gradients
            new_task_sections = []
            if self.opt["n_gradients"] > 0:
                gradients = self.get_gradients(
                    prompt[0],
                    task_section,
                    task,
                    gpt4,
                    texts,
                    labels,
                    preds,
                    seed,
                    curriculum,
                )
                new_task_sections = []
                for feedback, error_string in tqdm(
                    gradients, desc="applying gradients"
                ):
                    tmp = self.apply_gradient(
                        task_section,
                        error_string,
                        feedback,
                        self.opt["steps_per_gradient"],
                    )
                    new_task_sections += tmp

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc="mc samples"):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt["mc_samples_per_step"]
                    )
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            tmp_new_prompts = []
            for tmp in new_sections:
                if tmp[0] != [] and tmp[1] > 1.01:
                    tmp_new_prompts.append(
                        (prompt[0].replace(task_section, tmp[0][0]), tmp[1])
                    )

            if len(new_sections) > self.opt["max_expansion_factor"]:
                tmp_new_prompts = sorted(tmp_new_prompts, key=lambda x: x[1])
                tmp_new_prompts = tmp_new_prompts[: self.opt["max_expansion_factor"]]

            new_prompts += tmp_new_prompts

        new_prompts += prompts
        # NOTE: 按照perplexity进行排序
        new_prompts = sorted(new_prompts, key=lambda x: x[1])
        return new_prompts

    def score_candidates(self, prompts, task, gpt4, train_exs):
        """Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(
            prompts,
            train_exs,
            task,
            gpt4,
            scorer=self.scorer,
            rounds=self.opt["eval_rounds"],
            num_prompts_per_round=self.opt["eval_prompts_per_round"],
            samples_per_eval=self.opt["samples_per_eval"],
            max_threads=self.max_threads,
        )
        return evals
