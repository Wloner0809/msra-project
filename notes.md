### idea

**Task：用in-context learning来优化prompt**

首先这里的prompt肯定是含有soft prompt的，soft prompt是learnable的而hard prompt是不可学习的

利用in-context learning来优化，就是利用demonstration，目前了解到demonstration的数目对效果有影响

*Influence Function对data的选取？*

一个可能的setting是要用LLM自己迭代优化prompt？？？

Reading List:

1. When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations(ICLR)
2. P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks
3. Diverse Demonstrations Improve In-context Compositional Generalization
4. Data Curation Alone Can Stabilize In-context Learning
5. MPrompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension

---

### Prompt Tuning vs Prompt Engineering

来自一个[Youtube视频](https://www.youtube.com/watch?v=yu27PWzJI_Y)

Prompt Engineering是Hard prompt，通常是人工构建的

Prompt Tuning是Soft prompt，是AI生成的embedding，可解释性差

### Hard prompt vs Soft prompt

Hard prompt是Discrete Prompt，prompt是一个实际的文本字符串

> Hard prompts are manually handcrafted text prompts with discrete input tokens. ~ HuggingFace

Soft Prompt是Continuous prompt，直接在底层语言模型的embedding中进行描述

> Soft prompts are learnable tensors concatenated with the input embeddings that can be optimized to a dataset; the downside is that they aren’t human readable because you aren’t matching these “virtual tokens” to the embeddings of a real word. ~ HuggingFace

> “soft” prompts designed by an AI that outperformed human-engineered “hard” prompts. ~ [Source](https://arxiv.org/abs/2104.08691)

### Prefix Tuning vs Prompt Tuning vs P-tuning

1. Prefix Tuning与Prompt Tuning的区别：The prefix parameters are inserted in **all** of the model layers, whereas prompt tuning only adds the prompt parameters to the model input embeddings. The prefix parameters are also optimized by a separate feed-forward network (FFN) instead of training directly on the soft prompts because it causes instability and hurts performance
2. P-tuning(与Prefix Tuning的区别)：The prompt tokens can be inserted anywhere in the input sequence, and it isn’t restricted to only the beginning. The prompt tokens are only added to the input instead of adding them to every layer of the model. Introducing *anchor* tokens can improve performance because they indicate characteristics of a component in the input sequence

### Completion/Token Concept

**The inputs are called *prompts* and outputs are referred to as *completions*. **

LLMs take the input *prompts* and chunk them into smaller units called *tokens* to process and generate language. Tokens may include trailing spaces and even sub-words. This process is language dependent.