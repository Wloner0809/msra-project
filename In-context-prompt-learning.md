### Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing

> Submitted on **28 Jul 2021**

综述文章太长不看版$\rightarrow$[知乎讲解](https://zhuanlan.zhihu.com/p/396098543)，下面仅记录一些比较重要的综述内容

1. **四种范式：**

   ![](assets/p18.png)

   1. **Feature-Engineering(特征工程)：**纯有监督学习为主，需要一定规模的标注数据，然后学习模型参数，再基于模型对新的句子进行解码inference
   2. **Architecture-Engineering(架构工程)：**以设计新的神经网络模型为主的有监督学习
   3. **Objective-Engineering(目标工程)：**以设计新的预训练任务为代表
   4. **Prompt-Engineering(提示工程)：**比如**填空、前缀**等等，可以诱发/检索出大模型中所含有的实际任务所需要的

2. **Prompting Methods：**

   ![](assets/p19.png)

   1. **Prompting Function(提示函数)：**$f_{prompt}(\cdot)$负责把一个输入文本$x$变换为一个prompt $x'$，即为$x'=f_{prompt}(x)$

      * 首先应用一个“模板”，其中该模板应包含一个输入`slot[X]`和一个答案`slot[Z]`，`[Z]`最后会被映射给最后的输出y
      * 然后用输入文本$x$填充`slot[X]`

      > `[Z]`不在句子末尾的称为**cloze prompt(填空型提示)**；`[Z]`在末尾的称为**prefix prompt(前缀型提示)**

   2. **填充函数：**$f_{fill}(x',z)$负责用一个候选答案$z$来填充prompt $x'$中的`[Z]`，得到的prompt称为filled prompt

   3. **Answer Search(答案搜索)：**$\hat{z}=search_{z\in Z}P(f_{full(x',z);\theta})$，其中$P$即为PLM对prompt打分得到的概率。

   4. **Answer Mapping：**最后将得到的$\hat{z}$映射到任务定义的$y$

3. **代表性NLP任务：**

   ![](assets/p20.png)

   1. **Text CLS(文本分类任务)：**该任务还可细分成**sentiment(情感分析)、Topics(文本“主题”分类任务)、Intention(意图识别)**
   2. **Text-span CLS(文本片段分类任务)**
   3. **Text-pair CLS(文本对自然语言推理分类任务)**
   4. **Tagging(序列标注任务)**
   5. **Text Generation(文本生成任务)：**该任务可细分为**Summarization(总结)、Translation(翻译)**

4. **Prompt Engineering(提示工程)：**分为**discrete prompts(离散提示)、continuous prompts(连续提示)**

   1. **discrete prompts(离散提示)：**使用具体的words/tokens

      * **prompt mining(提示挖掘)**
      * **prompt paraphrasing(提示改述)**，例如$English\rightarrow Chinese\rightarrow English$这种
      * **Gradient-based Search(基于梯度的搜索)**
      * **Prompt Generation(提示生成)**
      * **Prompr Scoring(提示打分)**

   2. **continuous prompts(连续提示)：**基于embeddings来表示prompts，prompts拥有自己的参数可以微调

      * **Prefix-Tuning(前缀微调)**

        <img src="assets/p21.png" style="zoom:50%;" />

        在每个句子上加若干前缀，遇到新下游任务就修改prefix

      * **Tuning Initialized with Discrete Prompts(先离散后连续)**

      * **Hard-Soft Prompts Hybrid Tuning(离散+连续)**，例如将可微调的embeddings放入一个hard(离散的)prompt template

### Extensible Prompts for Language Models on Zero-shot Language Style Customization

> Submitted on **1 Dec 2022**, last revised **30 Nov 2023**

1. `Introduction`

   > 提出了**eXtensible Prompt(X-Prompt)**，将**imaginary words**$\rightarrow\widetilde{w}$注入**NL(natural language)**中构成**X-Prompt**，可以用于解决OOD robustness问题。同时提出**context-augmented learning(CAL)**的概念，更好地学习**imaginary words**$\rightarrow\widetilde{w}$，保证其general usability
   >
   > task的类型是language style customization

2. `eXtensible Prompt`

   * **X-Prompt：**$(w_{p_1},\cdots,w_{p_m})$，每个$w_{p_i}$可以来自NL vocabulary $V$或者extensible imaginary word vocabulary $\widetilde{V}$

   ![](assets/p1.png)

   * **Context-augmented learning：**假设**X-Prompt**为$$(w_{p_1},\cdots,\widetilde{w_{p_u}},\cdots w_{p_m})$$，那么学习**imaginary words**$\widetilde{w_{p_u}}$也即最大化$\log P(\vec{x}|w_{p_1},\cdots,w_{p_m})$，其中$\vec{x}$为training example$(w_{x_1},\cdots,w_{x_n})$
     * **Template augmentation：**给定$T$个**X-Prompt**，$\{(w_{p_1}^{(t)},\cdots,\widetilde{w_u},\cdots,w_{p_{m_t}}^{(t)})|1\leq t\leq T\}$，需要最大化$\frac{1}{T}\sum\limits_{t=1}^T\log P(\vec{x}|w_{p_1}^{(t)},\cdots,\widetilde{w_u},\cdots,w_{p_{m_t}}^{(t)})$

     * **Content augmentation：**向**X-Prompt**中注入an indicative keyword，对于keyword的处理如下图。

       对每个training example$\vec{x}$提取出keyword candidates$[w_k^1,\cdots,w_k^C]$，然后每个keyword插入到一个用于rank的prompt中选出最indicative的一个keyword，也即$w_k^\star=\arg\max\limits_{w_k^c}\log P(\vec{x}|\vec{r}(w_k^c))$，其中$\vec{r}(w_k^c)=(w_{p_1}^{(r)},\cdots,w_{p_{m_r}}^{(r)})$称为**ranking prompt template**

     ![](assets/p2.png)

3. `Experiments`

   * 实验主要是测试了两个task：**open-ended text generation**、**style transfer(rewriting)**，前一个任务是测试**X-Prompt**如何instruct语言模型生成user-specific语言，后一个任务是按要求转换语言的style(比如$impolite\rightarrow polite$)

     * open-ended text generation：

       1. 数据集：`Top 20 most followed users in Twitter social platform dataset`+`the Sentiment dataset5 from which we extract top 800 users’ (in total 68K) tweets (800-user dataset)`，同时剔除了test example中与training example具有相同indicative keyword的样本
       2. 基本配置：base model为`OPT-6.7b`，选用`Adam`优化器……
       3. 定量评估：选取的指标为`perplexity`和`accuracy`，实验结果如下图。**X-Prompt**在OOD上表现得更加好，而**Prompt tuning、X-Prompt(w/o CAL)**在ID上表现更好(因为它们focus on training examples)

       ![](assets/p3.png)

       4. 定性评估：手工构建了100个在training阶段没有见过的prompt，然后让两个人对LM生成结果在三个方面做评估：Content、Style、Overall，实验结果如下图

       ![](assets/p4.png)

     * style transfer：

       1. 数据集：`Entertainment (EM) subset of GYAFC (informal → formal)`+`POLITEREWRITE (impolite → polite)`
       2. 评测指标：`BLEU(Bilingual Evaluation Understudy), 该指标常用于机器翻译`用于评测生成结果和reference的lexical similarity、`accuracy`用于评测style appropriateness、`harmonic(H-) mean`和`geometric(G-) mean`用来作为overall performance

### Structured Prompting: Scaling In-Context Learning to 1,000 Examples

> Submitted on **13 Dec 2022**

1. `Introduction`

   > 提出了**Structured Prompting**，打破length的限制从而使得**In-Context Learning**可以用成千的examples训练。
   >
   > **In-Context Learning：**对于N-shot in-context learning，给定一个N labeled examples $D_{train}=\{(x_i,y_i)\}_{i=1}^N$，每个数据点都可用hand-crafted template $T$转化成一个demonstration $d_i=T(x_i,y_i)$。所有的demonstration可以被连接成$Z=d_1\oplus\cdots\oplus d_N$，对于每个test input $x_{test}$，都可以构造prompt为$Z$和$x_{test}$的连接。最终的输出结果为$\arg\max\limits_{c\in Y}P_{LM}(y^c|Z\oplus T(x_{test}))$，其中$Y$是所有可能的candidate

2. `Methods`

   * 

     ![](assets/p5.png)

     * **Group Context Encoding：**假设有$N$ demonstration examples，把这些examples随机分成$M$组 $\{Z_i\}_{i=1}^M$，每个group为$Z_i=d_{N_{i-1}+1}\oplus\cdots\oplus d_{N_i}$，其中$N_0=0$、$N_M=N$

       > Position Embedding：所有group采用right align从而保证它们有相同的max position index，因此所有group到test input有相同的距离。为了让test input对所有的examplar adjacent并pay equal attention，有两种方式：1. 使用left padding，i.e. pad tokens or space tokens 2.设置最大长度，从左边truncate examplar 

     * **Structured Prompting：**所有的examplar都喂进了Rescaled Attention，并连同test input一起喂进LM

       > Rescaled Attention：每一层都将所有examplar和test input的key、value连接起来，即$\hat{K}=[K_{Z_1},\cdots,K_{Z_M},K_x]$、$\hat{V}=[V_{Z_1},\cdots,V_{Z_M},V_x]$。计算attention
       >
       > 的公式为：
       >
       > ![](assets/p6.png)

3. `Experiments`

   1. 模型：`GPT-like(decoder-only Transformer)`，对于超大模型实验，选用`BLOOM-176B`
   2. 数据集：根据三个task：text classification、multi-choice、open-ended generation从而有对应的数据集

   > 整个实验测试了Model Size为**1.3B、6.7B、13B**在三个task `text classification`、`multi-choice`、`open-ended generation`，然后又在超大模型`BLOOM-176B`上测试了上面三个task，最后进行消融实验验证Prompt Length、Scaling Factor、Alignment Strategy的重要性

### How Does In-Context Learning Help Prompt Tuning

> Submitted on **22 Feb 2023**

1. `Introduction`

   > 本文主要比对了`PT(Prompt Tuning)`、`ICL(In-Context Learning)`、`IPT(Instruction Prompt Tuning)`从而来探究ICL对PT的影响
   >
   > **别的地方看到的：in-context examples主要是帮助model学习output label space和distribution of input text**

2. `Background`

   * `In-Context Learning`：在test input之前插入k个in-context input-output pairs，即为$Input_{ICL}=concat([X_{icl};Y_{icl}]_1^k;X_{test})$
   * `Prompt Tuning`：在test input $X_{test}$之前加入soft tunable prompt embeddings。一系列的tunable prompt embeddings用$E=\{e_1,\cdots,e_k\}$表示，那么$Input_{PT}=concat(E;X_{test})$。注意这里的$E$需要train，所以需要$X_{train}$、$Y_{train}$
   * `Instruction Prompt Tuning`：把soft prompts和hard in-context demonstrations连接在一起，即为$Input_{IPL}=concat(E;[X_{icl};Y_{icl}]_1^k;X_{test})$

3. `Experiments`

   1. 数据集：选用三种language generation tasks，即为`data-to-text generation`、`logic-to-text generation`、`semantic parsing`。不同任务有相应的数据集

   2. 模型：`BLOOM-1.1B`、`OPT-1.3B`、`GPT-2-XL-1.5B`

   3. **实验结论(这部分可以参考)**

      ![](assets/p11.png)

      * **ICL表现得比PT差**，这说明了对于类似OOD generation task，针对target task train一小部分参数是有价值的

      * **PT、IPT的表现难分伯仲**，取决于task类型和tunable parameter的数目等(work不work可能也还有数据集等因素)

      * **当demonstration跟test input类似的时候，IPT可以work**。这也就说明了similar demonstration对IPT的重要性

        ![](assets/p12.png)

      * **IPT在有更多的soft prompt tokens的情况下比PT表现得更稳定**

        ![](assets/p13.png)

      * **在有in-context demonstrations的情况下，Prompt embeddings对于新的task是transferable**

### Dynamic Prompting: A Unified Framework for Prompt Tuning

> Submitted on **6 Mar 2023**, last revised **27 May 2023**

1. `Introduction`

   > 提出了**DP(Dynamic Prompt)**，根据不同的instance/task来调整相对应prompt的**position、length、representation**(例如不同position相比传统的prefix/postfix可能会更好地捕捉语义信息)。**DP**的整体架构如下：
   >
   > ![](assets/p22.png)
   
2. `Methods`

   * **Unified View：**把prompt分为prefix和postfix两部分，对于输入$x\in R^{m\times d}$，query matrix是$Q=xW^W\in R^{m\times d}$ key matrix是$K=xW^K\in R^{m\times d}$ value matrix是$V=xW^V\in R^{m\times d_v}$。假设prompt的长度为$l$，那么$P=[P_1;P_2]$，其中$P_1\in R^{l_1\times d}, P_2\in R^{l_2\times d}$。最终的输入变成$x'=[P_1;x;P_2]\in R^{(l_1+m+l_2)\times d}$，新的key matrix变为$K'=x'W^K\in R^{(l_1+m+l_2)\times d}$ value matrix变为$V'=x'W^V\in R^{(l_1+m+l_2)\times d_v}$。通过矩阵分解：$Q'=\begin{bmatrix} Q_1\\ Q\\ Q_2 \end{bmatrix}, K'=\begin{bmatrix} K_1\\ K\\ K_2 \end{bmatrix}, V'=\begin{bmatrix} V_1\\ V\\ V_2 \end{bmatrix}$，其中$Q_1,K_1\in R^{l_1\times d}, Q_2,K_2\in R^{l_2\times d}, V_1\in R^{l_1\times d_v}, V_2\in R^{l_2\times d_v}$。因此对于输入$x'=[P_1;x;P_2]$来说，attention head module变为$Head=Attn([P_1;x;P_2]W^Q,[P_1;x;P_2]W^K,[P_1;x;P_2]W^V)=softmax(\frac{Q'K'^T}{\sqrt{d}})V'$省略$\sqrt{d}$也就可以化为$[softmax(P_1W^QK'^T)V';softmax(xW^QK'^T)V';softmax(P_2W^QK'^T)V']$。最终版：![](assets/p23.png)

   * **Dynamic Prompting：**

     1. **Dynamic Position：**用一个one-layer网络$POS_\theta$和Gumbel-Softmax优化得到针对不同task/instance的**dpos**参数，原始的prompt可以被分为$P=[P_{before},P_{after}]$，其中$P_{before}=[P_1,\cdots,P_{dpos}], P_{after}=[P_{dpos+1},\cdots,P_l]$，因此输入为$X'=[P_{before};X;P_{after}]$。$POS_\theta$的输出为$\alpha\in R^{l+1}$(这是一个二进制的串，$0$到$l$一共有$l+1$个可能的位置，每个位置对应的值为$0/1$)，这里选用Gumbel-Softmax方式处理保证可微，$logit=Gumbel-Softmax(POS_\theta(x),\tau)$，$logit$是二进制串，只有一个位置值为$1$。

        * *adap_ins_pos*：关注instance层面的position变化，需要添加$d\times (l+1)$个参数
        * *adap_pos*：关注task层面的position变化，需要添加$l+1$个参数

     2. **Dynamic Length：**用一个one-layer网络$LEN_\theta$和Gumbel-Softmax优化得到针对不同task/instance的$l^\star$参数，即为：

        <img src="assets/p24.png" style="zoom: 33%;" />

        但实际上model的输入长度要求是固定的，因此作者作了一个替代方案

     3. **Dynamic Vector：**使用prompt pools $Pool=\{P^{(1)},\cdots,P^{(k)}\}$生成dynamic prompts，train一个小的网络$P_{O_\theta}$来得到每个prompt $P^{(i)}$关于给定输入$x$的attention score，即为$P_{new}=\sum\limits_{i=1}^k \beta_i\cdot P^{(i)},\beta=softmax(P_{O_\theta}(x))$

     4. **Combination：**

        * *adap_ins_vec_pos*：同时更新dynamic position和prompt pool
        * *adap_pos_ins_vec*：先用dynamic position学到task层面的position，然后更新instance层面的prompt pool

3. `Experiments`

   1. 数据集：采用五个SuperGLUE数据集来测试模型的language understanding ability

   2. 实验结果：

      1. **Adaptive Position：**

         ![](assets/p25.png)

         可以看到总体趋势是*adap_ins_pos > adap_pos > fixed_pos*，**T5-LM-Large**模型work得最好可能说明**大模型更适合prompt tuning**

      2. **Adaptive Length：**

         <img src="assets/p26.png" style="zoom:50%;" />

         虽然有提升，但是提升不如**Adaptive Position**

      3. **Adaptive Prompt：**

         ![](assets/p27.png)

         可以看到在instance层面同时更新position和propmt pool效果并不好

### Pre-Training to Learn in Context

> Submitted on **16 May 2023**

1. `Introduction`

   > 提出了**PICL(Pre-training for In-Context Learning)**，旨在提高ICL能力的同时maintain泛化能力。**PICL**主要通过在数据侧做文章来提高ICL能力：用data automatically constructed from the general plain-text corpus来预训练模型，基于**很多paragraphs都包含“intrinsic tasks”**这样的假设。
   >
   > 具体来说，把相同类型intrinsic task的paragraph连接在一起构建一个**meta-training dataset**来预训练模型。用contrastive learning的方式训一个**Encoder**使得具有相同类型intrinsic task的paragraph在向量空间中具有类似的Embedding
   >
   > ![](assets/p7.png)

2. `Methods`

   * 对于Corpus $C$中的每个paragraph $z_0$，首先用retriever $R$寻找$k$个与$z_0$具有相同类型intrinsic task的paragraphs $\{z_1,z_2,\cdots,z_k\}$，然后被检索出来的paragraphs会被视为demonstrations与$z_0$连接在一起：$z_k\oplus z_{k-1}\oplus\cdots\oplus z_1\oplus z_0$，最后喂给模型

     * **Retriever：**核心是**task-semantics encoder** $E$(**其实就是对比学习的目标**)，作者将两个paragraph $z_0$和$z$的相似性定义为点乘$E(z_0)\cdot E(z)$

       1. `Encoder`：使用**RoBERTaBASE**作为base model，输出的vector是**输入paragraph的每个token的最后一层表示的平均**
       2. `Retrieval`：$R(z_0)=\{z_k,z_{k-1},\cdots,z_1\}=top-k_z(E(z_0)\cdot E(z))$，具体实现调用了**FAISS**库
       3. [Contrastive Learning](https://zhuanlan.zhihu.com/p/346686467)：不同task选用下游NLP数据集，最终形成了一个dataset $D$。对于$D$中的每个$z_0$，正样本$z^+$跟$z_0$有相同的task类型，负样本集合为$N(z_0)$，loss可以计算为$L(z_0,z^+,N(z_0))=-\log\frac{e^{E(z_0)\cdot E(z^+)}}{e^{E(z_0)\cdot E(z^+)}+\sum\limits_{z^-\in N(z_0)}e^{E(z_0)\cdot E(z^-)}}$。其中$z^+$是随机sample出来的，$N(z_0)$包括两种样本：1.**Easy Negatives：**$z_{easy}^-$ 2.**Hard Negatives：**$z_{hard}^+$

       ![](assets/p8.png)

     * **Data Construction：**对于每个$z_0\in C$，连接retrieved paragraphs$\{z_1,z_2,\cdots,z_k\}=R(z_0)$从而得到a pre-training instance$z_k\oplus\cdots\oplus z_0$。评价一个instance的informativeness：$s=\frac{-\sum\limits_{i=0}^k \log P(z_i)+\log P(z_k\oplus z_{k-1}\oplus\cdots\oplus z_0)}{|z_k\oplus z_{k-1}\oplus\cdots\oplus z_0|}$，其中$|\cdot|$是instance的长度，$P(\cdot)$是language modeling probability

     * **Pre-training：**计算了整个sequence的loss $L_{ICL}(\theta)=-\frac{1}{N}\sum\limits_{i=1}^N\log P(z_k^i\oplus z_{k-1}^i\oplus\cdots\oplus z_0^i;\theta)$，再添加一个language modeling loss $L_{LM}(\theta)$。最终的优化目标为$\min\limits_{\theta}\alpha L_{ICL}(\theta)+(1-\alpha)L_{LM}(\theta)$

3. `Experiments`

   1. 数据集：用`OPENWEBTEXT`、`WIKICORPUS`、`BOOKCORPUS`构建pre-training data，corpus $C$一共包括$80M$ paragraphs，对于每个paragraph找$k=20$ demonstrations并连接它们到$1024tokens$

   2. Baseline：

      * `VanillaICL`：用连接的training examples直接prompt PLM
      * `ExtraLM`：在原始的full document被分成paragraph之前进一步pre-train PLM
      * `Self-Sup`：设计了四个自监督的预训练目标 **Next Sentence Generation, Masked Word Prediction, Last Phrase Prediction, and Classification**
      * `MetaICL`：用人工标注的下游数据集[meta-train](https://zhuanlan.zhihu.com/p/136975128)模型

   3. 评估：

      * **Few-Shot Text Classification：**

        * `ExtraLM`有效，说明corpus的diversity很大；`metaICL`有效，说明meta-training对ICL的提升有作用；`Self-Sup`无效，说明训练时分类task受限的label space给模型输出带来了bias

        ![](assets/p9.png)

      * **Instruction Following：**测试模型的泛化性

        ![](assets/p10.png)

        * `PICL`比`MetaICL`在更多的task上表现得更好说明了**与直接在下游任务上微调相比，在intrinsic task上预训练更能提升ICL能力、泛化能力更强**。`PICL`在text generation等任务上表现更好而`MeatICL`在**“Yes/No”**问题上表现更好，说明在下游数据集训练会导致对某些label过拟合

      > 文章后面探究了**Retriever、Demonstration Number、Filtering、Full Documents、Data Amount、Data Comparison**的影响

### Efficient Prompting via Dynamic In-Context Learning

> Submitted on **18 May 2023**

1. `Introduction`

   > 提出了**DYNAICL(Dynamic In-Context Learning)**，为有效解决performance-efficiency trade-off问题提供了一种方案。model size和sample size是影响计算低效的两个原因，后者可以通过减少prompt长度实现，而prompt长度受到in-context learning使用demonstration数目的影响。因此该论文的核心是train一个meta controller来分配in-context demonstration的数目
   >
   > ![](assets/p14.png)

2. `Methodology`

   * **Meta Controller：**采用instruction-tuned model **FLAN-T5**作为base model，主要关注分类任务。训练分为**两阶段**
     * 第一阶段目标是train一个meta controller，可以使得**generalist model, like Chatgpt**生成**“good output”**的同时利用最少的in-context examples，输出所需的example数目$k$，即为下图(其中$t$是一个threshold)。![](assets/p15.png)
     * 第二阶段是利用**强化学习**来微调meta controller
   * **Dynamic In-Context Example Allocation：**考虑到实际的computation budget，假设总共有$K$ samples $N$ tokens 每个example的平均长度为$L$，平均分配的baseline为$\frac{N}{K\times L}$。**DYNAICL**的分配策略是$E(P)=[\beta\cdot(C(P)/\widetilde{C})\cdot N/(K\cdot L)]$，其中$C(P)$是meta controller的预测结果，$[\cdot]$代表取整操作，$\widetilde{C}$是所有examples的平均预测结果，$\beta$是token saving ration

3. `Experiments`

   1. 数据集：选用`a subset in the FLAN collection containing 30+ classification tasks`来训练meta controller，大部分数据集都是分类任务，用作训练；少部分数据集不是分类任务，用作评估。同时一些分类任务对应的数据集会作为unseen task来评估

   2. 模型：选用`ChatGPT`作为generalist model，`LLAMA-65B`作为unseen generalist model评估meta controller的泛化性

   3. Baseline：

      * **uniform baseline：**每个sample分配相同数目的in-context examples
      * **random baseline：**遵循高斯分布随机选取一定数量的in-context examples

   4. Preliminary：

      ![](assets/p16.png)

      根据上图可以看出大部分可能只需要很少的shots就可以work，且越多的shots效果提升并不明显

   5. 结果：

      1. 相同performance节省token，相同token在performance上表现更好
      2. 对于Unseen Generalist Model、Unseen Task都有很好的泛化性
      3. in-context examples的分布(target budget设为5)：<img src="assets/p17.png" style="zoom: 50%;" />

### Prompt Optimization via Adversarial In-Context Learning

> Submitted on **5 Dec 2023**, last revised **28 Feb 2024**
>
> Rejected by ICLR2024

