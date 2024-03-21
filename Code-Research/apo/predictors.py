from abc import ABC, abstractmethod
from liquid import Template

import utils


class GPT4Predictor(ABC):
    def __init__(self, opt):
        # 传入的opt是一个字典
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass


class BinaryPredictor(GPT4Predictor):
    categories = ["No", "Yes"]

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(
            text=ex["text"]
        )  # 用ex['text']替换prompt中的{{text}}
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1, temperature=self.opt["temperature"]
        )[0]
        pred = 1 if response.strip().upper().startswith("YES") else 0
        return pred
