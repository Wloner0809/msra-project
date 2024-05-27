# msra-project
~~This is a project using In-Context Learning to optimize the prompt.~~

**Topic**: Use LLM to automatically optimize the prompts.

##  :calendar: Timeline

|          Data           |                           Summary                            |
| :---------------------: | :----------------------------------------------------------: |
|       Before 4.15       | Research some papers related to **automatic-prompt-optimization** and **ICL** |
|    :city_sunset:4.15    |  Official opening of this project organized by **Nan Yang**  |
| :night_with_stars:4.16  | Two directions are proposed :one: (**High Priority**)Utilize the idea of **curriculum learning** to arrange the examplars from easy to difficult. This process can be automated by LLM(serve as **Difficulty Measurer + Training Scheduler**) :two: Design a new metric to select best examplars(**Influence Function/Perplexity/Mutual Information/Semantic Distance/Entropy**).Now the task is using llama2-70b to test this idea at a random dataset. |
| :night_with_stars:4.20  | Two optimization areas in apo codes: :one:errors are selected randomly:two:use curriculum learning to select minibatch​ |
|   :crescent_moon:4.22   | implement the idea and make some tests(maybe more result files will be pushed tomorrow) |
|   :city_sunrise:4.24    | test llama/mixtral/qwen model on liar dataset, record the results |
| :night_with_stars:4.25  | test llama/qwen model on ethos dataset, record the results. Next: **Design a new metric to select prompts(instead of using acc)、Deal with error strings in a more reasonable way** |
| :night_with_stars: 5.10 | Use perplexity to select prompts(add Influence Score later)  |
|        5.13-5.27        | test models on liar/sst2 datasets and record all the results. write reports and presentation ppt. |

##  :book: Paper List

### In-Context Learning
1. [Active Example Selection for In-Context Learning](https://arxiv.org/abs/2211.04486)
   * Submitted on 8 Nov 2022
2. [Structured Prompting: Scaling In-Context Learning to 1,000 Examples](https://arxiv.org/abs/2212.06713)
   * Submitted on 13 Dec 2022
3. [Pre-Training to Learn in Context](https://arxiv.org/abs/2305.09137)
   * Submitted on 16 May 2023

### Prompt Tuning/Engineering
1. [Extensible Prompts for Language Models on Zero-shot Language Style Customization](https://arxiv.org/abs/2212.00616)
   * Submitted on 1 Dec 2022, last revised 30 Nov 2023
2. [Dynamic Prompting: A Unified Framework for Prompt Tuning](https://arxiv.org/abs/2303.02909)
   * Submitted on 6 Mar 2023, last revised 27 May 2023
3. [When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations](https://arxiv.org/abs/2310.19698)
   * Submitted on 30 Oct 2023

### Use ICL to optimize prompt
1. [How Does In-Context Learning Help Prompt Tuning?](https://arxiv.org/abs/2302.11521)
   * Submitted on 22 Feb 2023
2. [Efficient Prompting via Dynamic In-Context Learning](https://arxiv.org/abs/2305.11170)
   * Submitted on 18 May 2023
3. [Better Zero-Shot Reasoning with Self-Adaptive Prompting](https://arxiv.org/abs/2305.14106)
   * Submitted on 23 May 2023
4. [PhaseEvo: Towards Unified In-Context Prompt Optimization for Large Language Models](https://arxiv.org/abs/2402.11347)
   * Submitted on 17 Feb 2024 

### LLM as Optimizer
1. [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910)
   * Submitted on 3 Nov 2022, last revised 10 Mar 2023
2. [Automatic Prompt Optimization with “Gradient Descent” and Beam Search](https://arxiv.org/abs/2305.03495)
   * Submitted on 4 May 2023, last revised 19 Oct 2023
3. [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
   * Submitted on 7 Sep 2023, last revised 7 Dec 2023
4. [PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization](https://arxiv.org/abs/2310.16427)
   * Submitted on 25 Oct 2023, last revised 7 Dec 2023
5. [Prompt Engineering a Prompt Engineer](https://arxiv.org/abs/2311.05661)
   * Submitted on 9 Nov 2023, last revised 19 Feb 2024
6. [Are Large Language Models Good Prompt Optimizers?](https://arxiv.org/abs/2402.02101)
   * Submitted on 3 Feb 2024
