# 代码解析

## OPRO

`prompt_utils.py`

1. `call_openai_server_single_prompt()`: 是最基础的函数, 之后所有函数调用openai api的时候都会用这个函数
2. `call_openai_server_func()`: 相当于`call_openai_server_single_prompt()`的集成版, 输入的是字符串列表

`optimize_instructions.py`

1. `main()`: 主要进行了以下操作
   * scorer model的参数配置
   * optimizer model的参数配置
   * 测试scorer/optimizer servers
   * 其他超参数的设置, 数据集的处理与读取
   * 最后调用`opt_utils.py`的`run_evolution()`

`opt_utils.py`

1. `gen_ins_and_score_pairs_substr()`: 产生instruction-score对
2. `gen_meta_prompt()`: 生成meta-prompt, 主要包括meta-instruction、之前的instruction-score对、可能含有的一些examplars
3. `run_evolution()`: 主要进行了以下操作
   * 评估初始的instructions
   * evolution过程(循环num_search_steps次)
     1. 生成新的instructions(有few-shot和非few-shot的情况)
     2. 在few-shot examplars上评估新的instructions
     3. 在few-shot examplars上评估旧的instructions
     4. 在training set上评估新生成的instructions
     5. 每eval_interval步对新生成的instructions评估

`eval_utils.py`

1. `evaluate_single_instruction()`: 主要进行了以下操作
   * 生成初始prompt用于评估
   * 第二次prompt用于更好extract answer
   * 提取预测与ground truth对比得到准确率