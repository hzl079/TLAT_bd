# LAT 中 Robust Overfitting 的 Motivation Validation 实验说明

## 0. 文档用途

这份文档用于交给 coding agent，帮助 agent 理解实验目标、实验边界和执行顺序。

当前任务不是实现 REALISTA 对抗训练，也不是直接开始最终研究实验，而是先用 LAT 官方已有的潜空间对抗训练流程，快速验证一个研究动机：

> 在 LLM 的 latent adversarial training 中，是否可能出现 robust overfitting？

Agent 必须先理解并检查现有代码，完成实验前准备工作。除非用户明确确认，否则不要直接大规模训练，也不要修改原始 LAT 仓库。

---

## 1. 研究背景与最终研究方向

最终研究方向是：

```text
REALISTA 语义约束攻击
        +
LAT-style latent adversarial training
        +
robust overfitting 分析与缓解
```

REALISTA 原本主要提供推理阶段的语义约束潜空间攻击；LAT 提供潜空间对抗训练的 min-max 训练框架。

但是，在正式实现 REALISTA 语义字典攻击之前，需要先验证研究动机是否合理：

```text
对抗训练过程中，模型训练集上的鲁棒性持续提高，
但测试集上的鲁棒性是否会在达到峰值后下降？
```

本次实验只使用 LAT，不使用 REALISTA。

---

## 2. 本次实验的准确定位

本实验名称：

```text
LAT 中 robust overfitting 的 motivation validation
```

本实验属于：

```text
快速预实验
单配置观察实验
研究动机验证
```

本实验不属于：

```text
REALISTA 对抗训练正式实验
robust overfitting 的首次发现
完整论文级别的消融实验
新的对抗训练方法设计
```

需要特别注意：

《Exploring Robust Overfitting for Pre-trained Language Models》已经在预训练语言模型的 embedding-space adversarial training 中研究过 robust overfitting。

因此，本实验不能把“发现 robust overfitting”作为创新性结论。

本实验要验证的是：

> 已有研究在预训练模型中的 robust overfitting 现象，是否也能在 LAT 的 LLM latent adversarial training 设置中被观察到。

如果观察到，结果用于支持后续 REALISTA 对抗训练研究的 motivation。

---

## 3. 本次实验的严格边界

### 3.1 必须使用

```text
LAT 官方代码仓库
LAT 官方训练流程
LAT 官方潜空间攻击
LAT 官方支持的模型配置
LAT 官方对应的数据集和任务
```

### 3.2 本次暂时不使用

```text
REALISTA 代码
REALISTA 数据集
REALISTA 语义改写
REALISTA latent dictionary
REALISTA PGD/PLD
语义方向约束
scaled-simplex 约束
Qwen2.5-7B
新的对抗训练框架
对照组
防御方法
```

不要把 REALISTA 的代码、数据或攻击逻辑混入本次实验。

### 3.3 不修改原始仓库

原始 LAT 仓库位置：

```text
/home/work/hzl/LAT
```

不要直接修改这个目录中的代码。

建议在独立项目目录中工作，例如：

```text
/home/work/hzl/lat_robust_overfitting_validation
```

如果 agent 发现 LAT 官方代码可以直接运行，优先通过独立配置、启动脚本或复制后的项目运行，不要破坏原始仓库。

原始 REALISTA 仓库：

```text
/home/work/hzl/REALISTA
```

本次实验不需要使用 REALISTA。

---

## 4. 模型选择原则

本次实验的第一原则是：

> 不为了追求更小模型而更换 backbone；优先使用 LAT 官方代码已经支持、配置已经验证过的模型。

此前考虑过 Llama3.2-3B，但如果 LAT 官方代码并不直接支持它，就不要为了节省参数量而引入新的模型适配工作。

Agent 需要先检查 LAT 仓库中的：

```text
模型加载代码
默认配置
启动脚本
notebook
README
Hugging Face 模型名称
模型架构适配逻辑
```

然后找出：

```text
LAT 官方实验中可直接运行的最小或最容易复现的 backbone
```

如果 LAT 官方已经有明确的默认模型和训练配置，优先使用该配置。

如果存在多个模型，选择原则如下：

1. 官方代码无需改动或只需最少配置修改；
2. 官方数据、任务和 loss 与该模型已经匹配；
3. 模型规模相对较小；
4. 模型和 tokenizer 已经在服务器上存在，或下载成本可接受；
5. 能够稳定运行 latent attack 和外层训练。

不要只根据模型名称判断是否兼容。必须检查实际代码。

---

## 5. 数据集与任务选择

本次实验优先沿用 LAT 官方数据集、任务和训练目标。

不要因为换模型或为了方便，擅自把任务改成：

```text
普通文本分类
新的问答任务
新的指令微调任务
REALISTA 的问题-回答任务
```

除非检查代码后发现原任务与所选 backbone 完全不兼容，否则保持 LAT 原任务不变。

Agent 需要确认：

```text
LAT 使用的具体数据集是什么？
训练集、验证集、测试集如何划分？
任务是安全训练、拒答训练、unlearning 还是其他生成任务？
模型的训练目标是什么？
robust metric 在 LAT 中如何定义？
```

如果官方数据量较大，为了快速验证可以使用固定子集，但必须满足：

```text
训练子集、验证子集、测试子集固定；
不同 epoch 不重新随机抽样；
训练集和评测集不重叠；
记录采样规则和随机种子；
最终报告中标明这是 pilot experiment。
```

建议的快速规模仅作为起点，不要强行套用：

```text
训练集：1,000–5,000 条
验证集：500 条左右
测试集：500–1,000 条
```

如果 LAT 官方数据本身不大，则直接使用完整数据。

---

## 6. 本次实验的训练方法

直接使用 LAT 官方的 latent adversarial training。

抽象流程如下：

```text
读取一个训练 batch
        ↓
固定模型参数
        ↓
在 LAT 指定的潜空间位置优化扰动
        ↓
得到 adversarial hidden state
        ↓
计算 latent adversarial loss
        ↓
更新模型参数
        ↓
进入下一个 batch
```

本次不修改 LAT 的核心攻击定义，包括：

```text
扰动位置
扰动参数化方式
扰动预算
PGD 步骤
投影方式
攻击目标
外层训练 loss
```

如果为了运行必须修复明显的兼容性 bug，agent 需要：

1. 先记录问题；
2. 说明为什么该修改不改变 LAT 方法定义；
3. 保留原始代码；
4. 在独立项目目录中修改；
5. 修改后先做小规模验证；
6. 把修改内容和影响写入实验日志。

---

## 7. 评测 robust overfitting 的核心思想

robust overfitting 不是看最终模型是否鲁棒，而是看训练过程中训练集和测试集鲁棒性的变化关系。

重点观察：

```text
训练集 robust loss 持续下降；
测试集 robust loss 先下降后上升。
```

等价地，如果使用性能指标：

```text
训练集 robust metric 持续改善；
测试集 robust metric 达到峰值后下降。
```

如果使用攻击成功率：

```text
训练集 attack success rate 持续下降；
测试集 attack success rate 先下降后上升。
```

其中攻击成功率越低通常表示鲁棒性越好，具体方向必须以 LAT 官方任务定义为准。

---

## 8. 为什么还要记录 clean loss

clean loss 不是判定 robust overfitting 的主要指标，但必须作为辅助指标记录。

需要它来区分：

### 8.1 robust overfitting

```text
train robust loss 持续下降；
test robust loss 后期上升；
test clean loss 基本稳定，或没有同步严重恶化。
```

### 8.2 普通过拟合或整体泛化下降

```text
train clean loss 继续下降；
test clean loss 也持续上升；
test robust loss 同时上升。
```

第二种情况不能简单地全部称为 robust overfitting，因为模型可能发生了整体任务泛化退化。

clean loss 还用于检查：

```text
模型是否正常收敛；
对抗训练是否严重损害 clean performance；
LAT 攻击是否真的参与训练；
训练过程是否出现数值异常。
```

---

## 9. 必须记录的指标

### 9.1 核心鲁棒指标

至少记录：

```text
train robust loss
validation robust loss
test robust loss
```

根据 LAT 任务，再记录官方定义的鲁棒任务指标，例如：

```text
train robust metric
validation robust metric
test robust metric
attack success rate
refusal rate
安全响应率
unlearning effectiveness
```

具体指标名称以 LAT 官方评测代码为准。

### 9.2 Clean 辅助指标

```text
train clean loss
validation clean loss
test clean loss
train clean metric
validation clean metric
test clean metric
```

### 9.3 训练诊断指标

尽可能记录：

```text
攻击前 loss
攻击后 loss
每一步 PGD 的 loss
潜空间扰动范数
训练 step time
GPU 显存峰值
学习率
梯度范数
是否出现 NaN 或 Inf
```

---

## 10. 测试 checkpoint 的攻击要求

这是本实验最重要的评测要求。

每个 epoch 保存 checkpoint 后，评测当前 checkpoint 时必须针对当前模型重新生成或重新优化潜空间攻击。

标准流程：

```text
保存 epoch-k checkpoint
        ↓
加载 epoch-k checkpoint
        ↓
对 train 子集重新运行 LAT latent attack
        ↓
对 validation 重新运行 LAT latent attack
        ↓
对 test 重新运行 LAT latent attack
        ↓
记录 clean 与 robust 指标
```

不能这样做：

```text
只在初始模型上生成一次扰动；
后续所有 checkpoint 都重复使用初始扰动。
```

原因是模型参数会持续变化，初始模型的扰动不一定仍然是后续模型的有效攻击。复用旧扰动可能高估后期鲁棒性，从而掩盖 robust overfitting。

如果 LAT 官方评测代码已经有当前 checkpoint 的攻击流程，优先复用它，但要确认它确实会重新优化攻击，而不是加载旧的扰动缓存。

---

## 11. 训练周期与 checkpoint

本实验必须覆盖测试鲁棒性达到峰值之后的训练阶段。

建议：

```text
随机种子：先使用 1 个
训练：先跑短程确认流程，再延长训练
checkpoint：每个 epoch 保存
评测：每个 epoch 评测当前 checkpoint
```

训练 epoch 数不能机械固定为 10 或 20。

如果在当前训练结束时：

```text
test robust loss 仍在下降；
test robust metric 仍在改善；
```

不能得出“没有 robust overfitting”的结论，只能说明训练还没有覆盖到潜在过拟合阶段。

建议先做：

```text
少量 step：检查代码；
1–3 epochs：检查训练和评测闭环；
足够长的训练：观察测试鲁棒性是否先好后坏。
```

---

## 12. robust overfitting 判定标准

当前实验只做单配置、单种子 pilot，因此结论要谨慎。

可以报告“观察到与 robust overfitting 一致的趋势”，至少需要满足：

```text
1. train robust loss 在后期仍然下降或保持改善；
2. test robust loss 先下降，随后持续多个 checkpoint 上升；
3. test robust metric 达到峰值后下降，
   或 test attack success rate 达到低点后上升；
4. 现象不是单个 checkpoint 的随机波动；
5. clean 指标被同时记录，可以判断是否存在整体泛化退化。
```

建议定义并报告：

```text
best checkpoint：
    validation robust loss 最低，
    或 validation robust metric 最好的 checkpoint

last checkpoint：
    预设训练终点的 checkpoint

robust overfitting gap：
    test robust loss(last) − test robust loss(best)
```

如果指标越高越好，则可以报告：

```text
robust metric gap：
    best test robust metric − last test robust metric
```

如果使用 ASR，则可以报告：

```text
ASR gap：
    last test ASR − best test ASR
```

checkpoint 选择应使用 validation set，test set只用于最终报告，不用于训练过程中的选择。

---

## 13. 预期结果与解释

### 情况 A：观察到 robust overfitting

可以得出：

> 在当前 LAT 模型、数据集、攻击预算和训练周期下，观察到了训练鲁棒性与测试鲁棒性分化的现象，该现象与 robust overfitting 一致。

这支持后续研究动机：

```text
如果普通 LAT 的潜空间对抗训练已经存在鲁棒泛化风险，
则有必要进一步研究 REALISTA 语义约束攻击下的训练行为。
```

不能说：

```text
首次发现 LLM 对抗训练中的 robust overfitting；
证明所有 LAT 配置都会出现 robust overfitting；
证明 REALISTA 对抗训练一定会出现 robust overfitting。
```

### 情况 B：没有观察到 robust overfitting

不能立即否定动机。先检查：

```text
训练是否足够长；
训练集是否太大；
攻击是否真正增加了 loss；
测试攻击是否对每个 checkpoint 重新生成；
测试攻击强度是否足够；
学习率是否过低或过高；
评测指标方向是否正确；
训练/测试划分是否固定；
是否存在数值异常。
```

如果确认流程正确但仍未出现，应如实记录：

> 在当前 LAT 配置下没有观察到明显 robust overfitting；这可能说明该现象依赖模型、任务、攻击预算或训练设置。

此结果也不排除 REALISTA 语义字典攻击产生不同的训练行为。

### 情况 C：出现不稳定趋势

如果只有单个 epoch 异常，或者不同重复运行方向不一致，不要直接判定为 robust overfitting。

应标记为：

```text
weak / unstable evidence
```

后续至少补充更多随机种子或更长训练。

---

## 14. Agent 的工作顺序

Agent 必须按以下顺序工作，不要一上来就修改训练代码。

### 阶段 1：检查仓库

检查：

```text
/home/work/hzl/LAT
```

重点寻找：

```text
README
requirements
训练入口
配置文件
模型加载
数据加载
latent attack
训练循环
checkpoint 保存
评测脚本
notebook
日志实现
```

输出一份代码结构说明，明确：

```text
哪个文件负责训练；
哪个文件负责内层攻击；
哪个文件负责数据；
哪个文件负责模型和 hook；
哪个文件负责评测；
当前官方支持哪些模型；
当前官方任务是什么；
当前是否已有 checkpoint 级别评测。
```

### 阶段 2：确认可复用的官方配置

确认：

```text
默认 backbone；
对应 tokenizer；
数据集；
训练目标；
攻击注入层；
攻击步数；
扰动预算；
batch size；
学习率；
训练 epoch；
评测指标。
```

优先选择不需要改动 LAT 核心逻辑的官方配置。

### 阶段 3：制定独立实验目录

不要直接修改原始仓库。

建议：

```text
/home/work/hzl/lat_robust_overfitting_validation
```

可以复制必要代码，或通过独立启动脚本引用原始仓库。

建议目录：

```text
lat_robust_overfitting_validation/
├── README.md
├── configs/
│   └── pilot.yaml
├── scripts/
│   ├── smoke_test.sh
│   ├── train_pilot.sh
│   └── evaluate_checkpoints.sh
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
├── data/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── plots/
└── logs/
```

如果直接复制 LAT 代码会造成大量重复，也可以保留官方仓库不动，在新目录中建立包装脚本和配置文件。

### 阶段 4：先做 smoke test

只运行少量 step，确认：

```text
模型能加载；
数据能加载；
latent attack 能运行；
攻击 loss 能计算；
模型参数能更新；
checkpoint 能保存；
checkpoint 能重新加载；
评测能针对当前 checkpoint 重新攻击；
没有 NaN/Inf；
没有明显显存泄漏。
```

### 阶段 5：运行短训练

运行 1–3 个 epoch 或少量固定 step，检查：

```text
train clean loss；
train robust loss；
validation clean loss；
validation robust loss；
test clean loss；
test robust loss；
```

确认日志格式和曲线可以生成。

### 阶段 6：运行长训练

只有 smoke test 和短训练正常后，才开始正式的 motivation pilot。

要求：

```text
每个 epoch 保存 checkpoint；
每个 checkpoint 重新运行当前模型的 latent attack；
固定训练/验证/测试子集；
记录完整指标；
绘制训练曲线。
```

---

## 15. Agent 在开始修改前必须向用户确认的问题

Agent 检查仓库后，如果以下信息在代码中无法明确确认，必须向用户提问，不要自行猜测：

```text
1. LAT 官方当前最小且可直接运行的 backbone 是哪个？
2. LAT 官方配置对应的具体数据集和训练任务是什么？
3. 服务器上该模型和数据是否已经存在？
4. 是否使用官方完整数据，还是固定子集？
5. 训练和测试攻击是否已经有现成评测接口？
6. LAT 官方代码是否需要兼容性修复？
7. 训练输出目录是否使用建议的新目录？
8. 单张卡或多卡启动方式是什么？
9. 训练周期是否需要覆盖官方配置之外的更长训练？
10. 当前任务中的 robust metric 应该采用什么方向解释？
```

如果这些问题都能从仓库和现有配置中确定，则告诉用户：

```text
没有发现阻塞性问题，可以开始实验准备。
```

在用户确认前，不要开始长时间训练。

---

## 16. 最终需要交付的实验准备结果

Agent 在实验准备阶段至少需要交付：

```text
1. LAT 仓库结构和训练入口说明；
2. 官方模型、数据集、任务和攻击配置说明；
3. 新建的独立项目目录；
4. 不修改原始 LAT 仓库的运行方案；
5. pilot 配置文件；
6. smoke test 启动方式；
7. checkpoint 级别评测启动方式；
8. 指标日志格式；
9. robust overfitting 判定脚本或分析脚本；
10. 训练和评测结果保存目录；
11. 仍需用户确认的问题清单；
12. 实验前风险和可能的兼容性问题。
```

---

## 17. 最终简化版实验定义

如果只保留最核心的信息，本实验就是：

```text
目标：
    快速验证 LAT 的 LLM 潜空间对抗训练中是否出现 robust overfitting。

模型：
    LAT 官方直接支持的、相对较小的 backbone。

数据：
    LAT 官方数据集。

任务：
    LAT 官方原始任务。

攻击：
    LAT 官方 latent adversarial attack。

训练：
    LAT 官方 adversarial training process。

REALISTA：
    本次完全不使用。

对照组：
    本次不设计额外对照组，只观察 LAT 单条训练曲线。

评测：
    每个 checkpoint 针对当前模型重新生成/优化 latent attack。

核心指标：
    train robust loss；
    test robust loss；
    test clean loss 作为辅助指标。

判定：
    train robust loss 后期仍下降，
    test robust loss 先下降后上升，
    并且该趋势持续多个 checkpoint。

实验定位：
    motivation validation，不是最终 REALISTA 实验，
    也不是 robust overfitting 的首次发现。
```

---

## 18. 给 Agent 的直接执行指令

请先阅读并理解本实验说明，然后只进行实验前准备：

1. 检查 `/home/work/hzl/LAT` 的代码结构和官方配置；
2. 找出官方最小且最容易直接运行的 backbone；
3. 确认其数据集、任务、loss、潜空间攻击和评测方式；
4. 不修改 `/home/work/hzl/LAT`；
5. 在独立目录中设计运行方案；
6. 检查是否已有 checkpoint 级别的重新攻击评测；
7. 如果缺少评测闭环，提出最小必要的补充方案；
8. 先运行或准备 smoke test，不要直接开始长训练；
9. 把发现的所有不确定项列出来；
10. 只有在所有关键问题得到确认后，才开始修改代码和搭建实验框架。

再次强调：

```text
本阶段不使用 REALISTA；
本阶段不实现 REALISTA 字典攻击；
本阶段不追求新的方法贡献；
本阶段只验证 LAT latent adversarial training 中
是否存在与 robust overfitting 一致的训练/测试分化现象。
```
