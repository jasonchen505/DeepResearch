# LLM & Agent 技术面试五类能力应对指南

> 基于 Tongyi DeepResearch 项目深度分析，针对五类面试能力的系统性准备
> 每个问题均从「项目代码/设计决策」出发，强调底层原理 → 局限性 → 改进方向

---

## 目录

- [第一类：底层原理深度理解](#第一类底层原理深度理解)
- [第二类：实验与方案验证能力](#第二类实验与方案验证能力)
- [第三类：问题定位与排查能力](#第三类问题定位与排查能力)
- [第四类：工程落地能力](#第四类工程落地能力)
- [第五类：业务与场景理解](#第五类业务与场景理解)

---

## 第一类：底层原理深度理解

> 核心要求：不是回答清楚概念，而是讲清楚**方法解决什么问题、存在哪些局限性、有哪些改进方法**

---

### Q1.1: ReAct 范式为什么这样设计？它解决的核心问题是什么？有什么局限？

**解决的核心问题**：

传统 LLM 有两个致命缺陷：
1. **知识封闭**：参数化知识有截止日期，无法获取实时信息
2. **推理幻觉**：在复杂推理中容易"编造"事实

ReAct 将 **推理链（Chain-of-Thought）** 和 **工具调用（Action）** 交替进行，让模型在每一步推理后都能用外部信息修正自己的判断。

**设计决策分析**（结合代码 `react_agent.py:120-226`）：

```
用户问题 → [Think → Act → Observe] × N轮 → <answer>
```

关键设计点：
- **`<tool_call>` 标签而非 function calling**：不依赖特定推理框架的 API 实现，兼容 OpenAI、vLLM、SGLang 等多种 serving 后端
- **stop token 设置为 `"\n<tool_response>"` 和 `"<tool_call>"`**（`react_agent.py:77`）：防止模型在一次生成中同时产出工具调用和 observation 的混合内容
- **observation 以 `user` role 注入**（`react_agent.py:179`）：保持 OpenAI API 兼容格式，且避免 assistant 自我引用导致的循环

**局限性**：

| 局限 | 具体表现 | 根因 |
|------|---------|------|
| **串行瓶颈** | 每次只能调用一个工具，多步推理线性叠加 | ReAct 本质是串行决策链 |
| **上下文膨胀** | 每轮 tool result 累积，100K+ tokens 很常见 | 没有主动的记忆管理机制 |
| **推理不可回退** | 模型无法"撤回"错误的工具调用 | 单向决策，没有 backtracking |
| **工具调用格式脆弱** | JSON 解析失败率高（`react_agent.py:175-176`有 try-catch） | 依赖模型生成结构化文本而非原生 function calling |

**改进方向**：

1. **并行工具调用**：如 ParallelMuse 论文提出的同时发起多个搜索查询
2. **主动上下文压缩**：如 ReSum 论文提出的历史信息摘要
3. **置信度引导**：如 BrowseConf 论文的中途终止策略
4. **原生 function calling**：用 vLLM 的 guided decoding 保证 JSON 格式正确

---

### Q1.2: GRPO 相比 PPO 解决了什么问题？为什么在 Agent 场景下 GRPO 更合适？

**PPO 在 Agent 场景的核心问题**：

1. **Critic Network 训练困难**：Agent 轨迹长度可达 100K+ tokens，训练一个准确的价值函数极其困难。而且 Agent 的 reward 非常稀疏（只有最终答案的 0/1），中间步骤没有监督信号
2. **内存开销巨大**：PPO 需要同时维护 Policy 和 Critic 两个大模型，对于 30B 参数的 MoE 模型，内存直接翻倍
3. **优势估计方差大**：在稀疏奖励下，GAE 的优势估计方差很高，导致训练不稳定

**GRPO 的设计原理**：

```
对每个 prompt x:
  采样 G 个 responses: {y_1, ..., y_G}  （DeepResearch 中 G=roll_out_count=3）
  计算 rewards: {r_1, ..., r_G}
  优势估计: A_i = (r_i - mean(r)) / std(r)   ← 组内归一化，无需 Critic
  策略梯度: ∇J = E[∇log π(y|x) · clip(ratio, 1-ε, 1+ε) · A]
```

**DeepResearch 的 GRPO 变体特化**（来自 README）：

1. **Token-level 策略梯度**：传统 RL 只在 sequence 末端计算 reward 并回传，所有 token 共享同一个 advantage。DeepResearch 在 token 级别计算梯度，让模型能区分"哪一步决策是关键的"
2. **Leave-one-out 优势估计**：每个样本的 baseline 是其余所有样本的均值，比简单 group mean 更稳定
3. **负样本选择性过滤**：如果一个 prompt 的所有 rollout 都失败（全为 0 reward），则过滤掉该 prompt 的训练样本，避免无信号梯度更新

**局限性与改进方向**：

- **采样效率低**：GRPO 每个 prompt 需要采样 G 个完整轨迹，Agent 轨迹很长，采样成本极高
  - 改进：异步采样 + 经验回放（但要注意 off-policy 偏差）
- **奖励设计单一**：目前只有 0/1 reward，无法区分"差一点成功"和"完全错误"
  - 改进：中间步骤的 process reward（但标注成本高）
- **探索不足**：on-policy 限制了探索范围
  - 改进：DAPO 中提出的 higher clip epsilon 或 entropy bonus

---

### Q1.3: MoE 架构在 Agent 场景下有什么特殊优势和挑战？

**为什么选择 MoE**：

DeepResearch 使用 Qwen3-30B-A3B：30.5B 总参数，每 token 只激活 3.3B。

| 优势 | Agent 场景的具体收益 |
|------|-------------------|
| 大容量低计算 | Agent 需要处理超长上下文（110K tokens），MoE 让每 token 的计算量降低 ~10x |
| Expert 分工 | 不同 expert 可能自然分工处理搜索、数学推理、语言生成等不同类型的任务 |
| 训练效率 | RL 阶段需要大量采样，MoE 的低计算量让采样成本可控 |

**挑战**：

1. **负载均衡**：如果某些 expert 被过度使用，会导致计算瓶颈
2. **Expert 坍缩**：训练中可能出现所有 token 都路由到少数 expert 的情况
3. **推理调度复杂**：在 vLLM 部署中，MoE 的 expert 并行需要特殊的调度策略

**面试加分回答**：可以提到 DeepSeek-MoE 的 Shared Expert + Routed Expert 设计，以及如何在 Agent 场景下优化 expert 路由。

---

### Q1.4: 为什么 Visit 工具需要两阶段设计（Jina + LLM 摘要）？直接返回网页内容不行吗？

**两阶段设计分析**（`tool_visit.py:179-254`）：

```
原始网页 → [Jina Reader] → 原始HTML/文本 → [LLM摘要提取] → 结构化信息(rational/evidence/summary)
```

**为什么不能直接返回原始内容**：

1. **信息过载**：一个网页可能包含 100K+ tokens 的内容（导航栏、广告、评论等），直接注入上下文会迅速耗尽 context window
2. **信噪比低**：原始网页中与用户 goal 相关的信息可能只占 1%-5%
3. **格式不统一**：不同网页的 HTML 结构差异巨大，模型难以一致地解析

**为什么用另一个 LLM 而非规则提取**：

- 规则提取（如 BeautifulSoup）只能基于 DOM 结构提取文本，无法理解语义
- LLM 能理解用户 goal 并做定向信息抽取，输出 rational（定位）、evidence（证据）、summary（总结）三层结构

**局限性**：

| 问题 | 代码体现 | 影响 |
|------|---------|------|
| **摘要幻觉** | Summary Model 可能生成原文中没有的信息 | 传递错误信息给 Agent |
| **延迟叠加** | 每次 visit = Jina调用 + LLM调用，延迟 ~10-30s | 整体推理时间大幅增加 |
| **摘要失败回退** | `tool_visit.py:202-221` 有截断重试逻辑，但最终失败时返回空模板 | Agent 收到无用信息 |
| **Token 计数不精确** | `truncate_to_tokens` 用 tiktoken 而非模型原生 tokenizer | 截断位置可能不准确 |

**改进方向**：
1. 用小模型（如 1.5B）专门做摘要，降低延迟
2. 引入 retrieval-based 方法，先用 embedding 检索相关段落再摘要
3. 对摘要结果做事实一致性校验

---

### Q1.5: Agentic CPT（持续预训练）相比直接 SFT 有什么本质区别？为什么需要 CPT？

**问题背景**：

传统 Agent 训练流程：预训练 → SFT → RL。但直接 SFT 存在以下问题：

1. **分布偏移**：预训练数据是自然文本，SFT 数据是 ReAct 格式的交互轨迹，两者分布差异大
2. **灾难性遗忘**：SFT 可能损害模型的通用能力
3. **能力天花板**：SFT 只能学习已有的成功轨迹，无法获得新的基础知识

**Agentic CPT 的设计思路**（AgentFounder 论文）：

```
预训练模型 → [32K CPT: 基础Agent能力] → [128K CPT: 长上下文能力] → SFT → RL
```

CPT 阶段的数据是**大规模的 Agent 交互数据**，但训练目标和预训练一样（next token prediction），这使得模型在**保持通用能力的同时**逐步获得 Agent 能力。

**关键创新**：
- **FAS (Planning Action Synthesis)**：从 QA 数据自动生成规划-行动序列
- **HAS (Decision-Making Action Synthesis)**：将轨迹重构为多步决策过程
- **开放世界记忆**：将持续更新的数据流转化为训练数据源

**面试深挖**：CPT 数据的构造方式决定了 Agent 的能力上限。如果 CPT 数据中搜索查询的质量不高，模型学到的就是"如何发出低质量搜索"，这比没有 CPT 还糟糕。

---

### Q1.6: Token-level 策略梯度 vs Trajectory-level 策略梯度，技术细节和实际影响？

**Trajectory-level（传统做法）**：

```python
reward = 1.0 if answer_correct else 0.0
# 所有 token 共享同一个 advantage
for token in trajectory:
    loss += -log_prob(token) * reward
```

问题：无法区分哪些 token 贡献了成功/失败。在一个 50 步的 Agent 轨迹中，可能只有第 30 步的关键搜索查询决定了成败，但所有 50 步的 token 都得到了相同的 reward。

**Token-level（DeepResearch 的做法）**：

- 在每个 token 位置计算策略梯度
- 结合 leave-one-out 估计，每个 token 的 advantage 不同
- 理论上能让模型学到"在哪个 step 做出了关键决策"

**实际挑战**：
1. **Credit Assignment 问题**：即使 token-level，reward 仍然是在轨迹末端才获得的，如何将 reward 归因到中间 token？
2. **方差问题**：token-level 梯度的方差比 trajectory-level 更大
3. **计算开销**：需要在每个 token 位置计算梯度，计算量增加

---

## 第二类：实验与方案验证能力

> 核心要求：面试官不仅关注你做了什么，更关注**怎么证明它是有效的**，追问实验细节

---

### Q2.1: 如何验证 ReAct 推理范式比纯 CoT 或纯工具调用更好？

**实验设计方案**：

| 方法 | 描述 | 对比维度 |
|------|------|---------|
| Pure CoT | 只靠模型推理，不调用工具 | 准确率、幻觉率 |
| Pure Tool | 直接调用工具，不做推理 | 搜索效率、信息利用率 |
| ReAct | 推理+工具交替 | 综合性能 |

**评估指标**：
- **Pass@1 / Pass@3**：单次/多次尝试的成功率
- **工具调用效率**：平均每个问题的工具调用次数
- **推理质量**：人工评估 reasoning chain 的逻辑性
- **终止原因分布**：正常 answer / 超时 / token 溢出 / 超出调用次数

**关键实验细节**（面试追问点）：

1. **如何保证公平对比？**
   - 相同的模型、相同的温度参数、相同的工具集
   - 相同的评估数据集（如 GAIA 的 test set）
   - 相同的 rollout 次数（Pass@3 需要多次采样）

2. **如何处理随机性？**
   - 多次运行取平均（至少 3 次）
   - 报告置信区间而非单点估计
   - 控制温度参数（`temperature=0.6, presence_penalty=1.1`）

3. **如何做消融实验？**
   - 去掉某个工具（如去掉 Scholar），看性能下降多少
   - 改变 `presence_penalty` 的值，看探索程度的影响
   - 限制最大 LLM 调用次数（如从 100 降到 50），看性能衰减曲线

---

### Q2.2: 如何验证 GRPO 比 PPO 在 Agent RL 中更有效？

**实验设计**：

```python
# 同一基座模型，分别用 PPO 和 GRPO 训练
models = {
    "SFT_baseline": sft_model,
    "PPO_agent": train_ppo(sft_model, agent_data),
    "GRPO_agent": train_grpo(sft_model, agent_data),
    "GRPO_token_level": train_grpo_token_level(sft_model, agent_data),
}

# 在多个 benchmark 上评估
benchmarks = ["BrowseComp", "GAIA", "WebWalkerQA", "FRAMES"]
```

**关键对比指标**：

| 指标 | PPO | GRPO | 说明 |
|------|-----|------|------|
| 训练稳定性 | 需要精细调参 | 更稳定 | Critic 的方差影响 |
| 内存占用 | ~2x 模型大小 | ~1x | 无需 Critic |
| Pass@1 | X% | Y% | 最终性能 |
| 训练时间 | T1 | T2 | 采样效率 |

**面试追问点**：

1. **GRPO 的采样数量 G 如何选择？**
   - G 太小（如 2）：组内方差估计不准
   - G 太大（如 10）：采样成本过高
   - DeepResearch 默认 `roll_out_count=3`，这是一个经验性的平衡点

2. **如何评估训练稳定性？**
   - 观察 reward 曲线的方差
   - 监测 KL divergence（策略偏离参考策略的程度）
   - 检查是否有 reward hacking（模型学到投机取巧的策略）

3. **Leave-one-out 优势估计的具体实现**：
   ```python
   # 对于 G=3 的情况
   rewards = [r1, r2, r3]
   # 样本1的 baseline = mean(r2, r3)
   # 样本2的 baseline = mean(r1, r3)
   # 样本3的 baseline = mean(r1, r2)
   advantages = [(r1 - mean(r2,r3)), (r2 - mean(r1,r3)), (r3 - mean(r1,r2))]
   advantages = normalize(advantages)  # 标准化
   ```

---

### Q2.3: 如何评估数据合成方法的质量？WebShaper 的形式化方法比传统方法好在哪？

**评估维度**：

1. **多样性评估**：
   - 问题类型的分布（检索、推理、多跳等）
   - 知识领域的覆盖度
   - 难度级别的分布

2. **质量评估**：
   - 人工标注：随机抽样 100 条，评估问题清晰度、答案正确性
   - 自动评估：用 LLM 做一致性检查（question-answer consistency）
   - Reasoning shortcut 检测：检查是否有多余的捷径

3. **训练效果评估**：
   - 用合成数据训练模型，在标准 benchmark 上评估
   - 对比不同合成方法的训练数据（相同数据量）

**WebShaper vs 传统方法的实验设计**：

```
传统方法：
  1. 从网页中提取实体和关系
  2. 基于模板生成 QA 对
  3. 过滤低质量样本

WebShaper 方法：
  1. 建立 KP 形式化框架
  2. 基于形式化约束生成 QA
  3. Layer-wise 结构保证无 reasoning shortcut
```

**对比指标**：
- GAIA benchmark 上的 Pass@1
- 模型在合成数据上的过拟合速度
- 人工评估的 reasoning shortcut 比例

---

### Q2.4: 如何设计 A/B 实验验证 Agent 的线上效果？

**线上 A/B 实验设计**：

1. **分流策略**：
   - 按用户随机分流（50/50 或 90/10）
   - 控制变量：同一问题只分配到一个版本

2. **评估指标**：
   - **用户满意度**：用户对回答的评分（1-5分）
   - **任务完成率**：用户是否得到所需信息
   - **响应时间**：从提问到给出最终答案的时间
   - **工具调用次数**：平均每个问题的工具调用数
   - **成本**：API 调用费用、计算资源

3. **统计显著性**：
   - 至少运行 1000 个问题
   - 使用 bootstrap 或 t-test 计算 p-value
   - 关注 effect size 而非仅 p-value

---

## 第三类：问题定位与排查能力

> 核心要求：模型上线后能力下降、系统变慢、结果异常，怎么排查？强调**优化思路与解决方案**

---

### Q3.1: 上线后发现 Agent 的 Pass@1 突然从 60% 降到 40%，如何排查？

**系统性排查思路**：

```
┌──────────────────────────────────────────────┐
│              性能下降排查流程                   │
├──────────────────────────────────────────────┤
│  1. 确认数据一致性                             │
│     → 评估数据集是否被修改？                    │
│     → 数据预处理逻辑是否变更？                  │
├──────────────────────────────────────────────┤
│  2. 检查模型服务                               │
│     → vLLM 版本是否更新？                      │
│     → 模型权重是否被覆盖？                     │
│     → 量化精度是否变化？                       │
├──────────────────────────────────────────────┤
│  3. 检查工具链                                 │
│     → Serper API 是否限流/变更？               │
│     → Jina API 是否稳定？                     │
│     → SandboxFusion 是否正常？                 │
├──────────────────────────────────────────────┤
│  4. 检查推理配置                               │
│     → temperature/top_p 是否被修改？           │
│     → presence_penalty 是否变化？              │
│     → max_tokens 是否变化？                    │
├──────────────────────────────────────────────┤
│  5. 分析失败模式                               │
│     → 终止原因分布变化？（超时/格式错误/...）   │
│     → 哪些类型的问题性能下降最多？              │
│     → 工具调用的成功率是否变化？                │
└──────────────────────────────────────────────┘
```

**具体排查步骤**（结合代码）：

**Step 1：检查终止原因分布**

```python
# 从输出 JSONL 中统计 termination 类型
from collections import Counter
terminations = [item['termination'] for item in results]
print(Counter(terminations))
# 正常情况：'answer' 占多数
# 异常情况：'exceed available llm calls' 或 'vllm server error' 占比突增
```

**Step 2：检查工具调用成功率**

```python
# 分析 messages 中的 tool result
for item in results:
    for msg in item['messages']:
        if msg['role'] == 'user' and '<tool_call>' in msg.get('content', ''):
            result = msg['content']
            if 'Failed' in result or 'Error' in result:
                print(f"Tool failed for question: {item['question']}")
```

**Step 3：检查模型输出质量**

```python
# 对比新旧模型在同一问题上的输出
for question in sample_questions:
    old_output = old_model.generate(question)
    new_output = new_model.generate(question)
    if old_output != new_output:
        print(f"差异问题: {question}")
        print(f"旧输出: {old_output[:200]}")
        print(f"新输出: {new_output[:200]}")
```

**Step 4：检查外部 API 状态**

```python
# 测试 Serper API
import requests
response = requests.post("https://google.serper.dev/search",
    json={"q": "test query"},
    headers={"X-API-KEY": SERPER_KEY})
print(response.status_code, response.json().keys())
```

**常见原因和解决方案**：

| 原因 | 表现 | 解决方案 |
|------|------|---------|
| API 限流 | 工具调用失败率上升 | 增加重试、降低并发、升级 API 套餐 |
| 模型权重损坏 | 输出乱码或重复 | 重新下载模型、校验 hash |
| 温度参数变更 | 输出随机性增大 | 检查配置文件、回滚配置 |
| 评估数据泄露 | 训练集和测试集重叠 | 检查数据去重逻辑 |

---

### Q3.2: 系统上线后推理速度突然变得很慢，如何定位瓶颈？

**性能瓶颈分析框架**：

```
┌─────────────────────────────────────────────┐
│           推理延迟分解                        │
├─────────────────────────────────────────────┤
│  总延迟 = LLM推理延迟 + 工具调用延迟          │
│         + 上下文构建延迟 + 网络延迟            │
├─────────────────────────────────────────────┤
│  LLM推理延迟:                                │
│    - 首 token 延迟 (TTFT)                    │
│    - 解码速度 (tokens/s)                     │
│    - 队列等待时间                             │
├─────────────────────────────────────────────┤
│  工具调用延迟:                                │
│    - Search: ~1-3s (Serper API)              │
│    - Visit: ~10-30s (Jina + LLM摘要)         │
│    - Python: ~5-50s (SandboxFusion)          │
└─────────────────────────────────────────────┘
```

**排查步骤**：

**Step 1：加日志计时**

```python
import time

# 在 react_agent.py 的 _run 方法中
start = time.time()
content = self.call_server(messages, planning_port)
llm_time = time.time() - start
print(f"LLM call: {llm_time:.2f}s")

if '<tool_call>' in content:
    start = time.time()
    result = self.custom_call_tool(tool_name, tool_args)
    tool_time = time.time() - start
    print(f"Tool call ({tool_name}): {tool_time:.2f}s")
```

**Step 2：检查 vLLM 服务状态**

```bash
# 检查 GPU 利用率
nvidia-smi

# 检查 vLLM 日志中的排队情况
tail -f /var/log/vllm/server.log | grep "queue"

# 检查请求延迟分布
curl http://localhost:6001/metrics | grep request_latency
```

**Step 3：检查并发配置**

```python
# run_multi_react.py 中的并发设置
max_workers = args.max_workers  # 默认20
# 如果 max_workers 太大，会导致 vLLM 排队严重
# 如果太小，GPU 利用率不足
```

**Step 4：检查上下文长度**

```python
# 在 react_agent.py 中
token_count = self.count_tokens(messages)
print(f"round: {round}, token count: {token_count}")
# 如果 token_count 接近 110K，每次推理都会很慢
```

**常见瓶颈和解决方案**：

| 瓶颈 | 原因 | 解决方案 |
|------|------|---------|
| GPU 利用率低 | 并发不足 | 增加 max_workers |
| 排队严重 | 并发过多 | 减少 max_workers 或增加 GPU |
| Visit 延迟高 | Jina API 慢或 LLM 摘要慢 | 增加超时、用更小的摘要模型 |
| Token 数膨胀 | 没有上下文管理 | 引入主动摘要机制 |
| 网络延迟 | API 调用慢 | 使用缓存、CDN |

---

### Q3.3: 实验发现 RL 训练后模型性能反而下降了，如何分析？

**排查思路**：

**Step 1：检查 Reward 曲线**

```python
# 如果 reward 曲线在上升，但 benchmark 性能在下降
# → 可能是 reward hacking
# 模型学会了在 reward function 上投机取巧，而非真正解决问题
```

**Step 2：检查 KL Divergence**

```python
# KL(π_RL || π_SFT) 如果过大，说明模型偏离太远
# 可能需要调整 KL penalty 或降低学习率
```

**Step 3：分析 RL 训练数据**

```python
# 检查正负样本比例
positive = sum(1 for r in rewards if r > 0)
negative = sum(1 for r in rewards if r == 0)
print(f"Positive: {positive}, Negative: {negative}")
# 如果负样本过多，训练信号会被噪声淹没
```

**Step 4：检查评估一致性**

```python
# RL 训练中的评估和最终 benchmark 评估是否一致？
# 例如：训练时用简单匹配，评估时用 LLM judge
# → 可能存在评估标准不一致的问题
```

**常见原因**：

| 原因 | 诊断方法 | 解决方案 |
|------|---------|---------|
| Reward Hacking | 分析高 reward 但错误的样本 | 改进 reward function |
| 过拟合 | 训练 reward 上升但验证下降 | 减少训练步数、增加正则 |
| 分布偏移 | RL 数据和评估数据分布不同 | 用 on-policy 数据 |
| 学习率过大 | KL divergence 急剧上升 | 降低学习率 |
| 负样本过多 | 正样本比例 < 10% | 过滤负样本、调整采样策略 |

---

### Q3.4: Visit 工具的摘要质量不稳定，如何排查和优化？

**问题表现**：
- 有时摘要准确且完整
- 有时摘要丢失关键信息
- 有时摘要包含幻觉内容

**排查步骤**：

**Step 1：保存原始数据**

```python
# 在 tool_visit.py 中增加日志
def readpage_jina(self, url, goal):
    content = self.html_readpage_jina(url)
    
    # 保存原始网页内容
    with open(f"log/raw_{hash(url)}.txt", "w") as f:
        f.write(content)
    
    # 保存摘要结果
    summary = summary_page_func(messages)
    with open(f"log/summary_{hash(url)}.json", "w") as f:
        json.dump({"url": url, "goal": goal, "summary": summary}, f)
```

**Step 2：对比分析**

```python
# 对比原始内容和摘要
for log_file in log_files:
    raw = read_raw(log_file)
    summary = read_summary(log_file)
    
    # 检查关键信息是否在摘要中
    key_entities = extract_entities(raw)
    summary_entities = extract_entities(summary)
    missing = key_entities - summary_entities
    if missing:
        print(f"Missing entities: {missing}")
```

**Step 3：评估摘要模型**

```python
# 用不同的摘要模型对比
models = ["gpt-4o-mini", "qwen-7b", "qwen-72b"]
for model in models:
    summary = call_summary_model(model, content, goal)
    score = evaluate_summary(summary, ground_truth)
    print(f"{model}: {score}")
```

---

## 第四类：工程落地能力

> 核心要求：理论可行不代表工程可行，强调**理论结合实际的生产落地**

---

### Q4.1: DeepResearch 的推理系统是如何部署的？为什么选择这种架构？

**部署架构分析**（`run_react_infer.sh`）：

```bash
# 8 个 GPU 各自运行独立的 vLLM 实例
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --port 6001 &
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH --port 6002 &
...
CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH --port 6008 &
```

**为什么选择独立 serving 而非 Tensor Parallel？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| **独立 Serving**（当前方案） | 灵活、故障隔离、无通信开销 | 每个 GPU 的 KV Cache 独立，无法共享 |
| **Tensor Parallel** | 单请求延迟低、大模型支持 | 通信开销、一个 GPU 故障影响全部 |

**选择独立 Serving 的原因**：
1. **Agent 推理的特点**：每个 Agent 的推理是独立的，不需要跨 GPU 通信
2. **负载不均**：不同 Agent 的推理时长差异很大，独立 serving 可以灵活调度
3. **故障隔离**：一个 vLLM 实例崩溃不影响其他实例

**Sticky Port Assignment 设计**（`run_multi_react.py:117-140`）：

```python
# 同一问题的所有 rollout 分配到同一端口
question_to_ports = {}
for item in items:
    if question not in question_to_ports:
        planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
        question_to_ports[question] = planning_port
```

**为什么需要 Sticky Assignment？**
- 同一问题的多次 rollout 共享相似的上下文前缀
- 分配到同一端口可以利用 vLLM 的 prefix caching
- 避免不同端口的 KV Cache 重复计算

---

### Q4.2: 如何保证系统在生产环境中的稳定性？

**稳定性保障措施**：

**1. 多层重试机制**

```python
# react_agent.py:70-110 - LLM 调用重试
for attempt in range(max_tries):  # max_tries=10
    try:
        content = call_server(msgs, port)
        if content and content.strip():
            return content
    except (APIError, APIConnectionError, APITimeoutError):
        sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
        sleep_time = min(sleep_time, 30)
        time.sleep(sleep_time)

# tool_visit.py:143-167 - Jina 读取重试
for attempt in range(max_retries):  # max_retries=3
    response = requests.get(url, timeout=50)
    if response.status_code == 200:
        return response.text

# tool_python.py:76-109 - Python 执行重试
for attempt in range(8):
    endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)  # 多端点容错
    code_result = run_code(...)
```

**2. 超时控制**

```python
# react_agent.py:140 - Agent 级别超时
if time.time() - start_time > 150 * 60:  # 2.5小时
    return 'No answer found after 2h30mins'

# react_agent.py:186 - Token 级别超时
max_tokens = 110 * 1024
if token_count > max_tokens:
    # 强制给出答案

# tool_visit.py:84 - 工具级别超时
if time.time() - start_time > 900:  # 15分钟
    return timeout_template
```

**3. 断点续跑**

```python
# run_multi_react.py:91-108
for rollout_idx in range(1, roll_out_count + 1):
    processed_queries = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if "question" in data and "error" not in data:
                    processed_queries.add(data["question"].strip())
```

**4. 错误分类和记录**

```python
# run_multi_react.py:196-225
except concurrent.futures.TimeoutError:
    error_result = {"error": "Timeout (>1800s)", "prediction": "[Failed]"}
except Exception as exc:
    error_result = {"error": f"Future resolution failed: {exc}", "prediction": "[Failed]"}
```

---

### Q4.3: 如何设计 Agent 系统的监控和告警？

**监控指标体系**：

```python
# 推荐的监控指标
metrics = {
    # 性能指标
    "pass_at_1": "Pass@1 成功率",
    "avg_latency": "平均推理延迟",
    "avg_tool_calls": "平均工具调用次数",
    "avg_token_count": "平均 token 使用量",
    
    # 稳定性指标
    "error_rate": "错误率",
    "timeout_rate": "超时率",
    "tool_failure_rate": "工具调用失败率",
    
    # 成本指标
    "api_cost": "API 调用费用",
    "gpu_utilization": "GPU 利用率",
    "token_cost": "Token 消耗量",
}
```

**告警规则设计**：

```yaml
alerts:
  - name: "Pass@1 下降"
    condition: "pass_at_1 < 0.4"
    window: "1h"
    severity: "critical"
    
  - name: "推理延迟过高"
    condition: "avg_latency > 300s"
    window: "30m"
    severity: "warning"
    
  - name: "工具失败率过高"
    condition: "tool_failure_rate > 0.3"
    window: "15m"
    severity: "critical"
    
  - name: "API 费用超预算"
    condition: "api_cost > 1000"
    window: "24h"
    severity: "warning"
```

**日志设计**：

```python
# 每次 Agent 推理的结构化日志
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "question": question,
    "model": model_name,
    "rounds": round,
    "token_count": token_count,
    "tool_calls": tool_call_log,  # [{name, args, result_length, latency}]
    "prediction": prediction,
    "termination": termination,
    "latency": total_time,
    "error": error_message if error else None,
}
```

---

### Q4.4: 如何实现 Agent 系统的数据回滚？

**数据版本管理**：

```
eval_data/
├── v1/
│   ├── questions.jsonl
│   └── file_corpus/
├── v2/
│   ├── questions.jsonl
│   └── file_corpus/
└── current -> v2/  # 软链接指向当前版本
```

**模型版本管理**：

```
models/
├── v1/
│   └── model_weights/
├── v2/
│   └── model_weights/
└── current -> v2/
```

**回滚流程**：

```bash
# 1. 发现问题
python monitor.py --check-metrics --threshold 0.4

# 2. 回滚模型
ln -sfn models/v1 models/current

# 3. 回滚数据
ln -sfn eval_data/v1 eval_data/current

# 4. 重启服务
bash restart_services.sh

# 5. 验证回滚
python evaluate.py --model current --dataset current
```

---

### Q4.5: 如何优化推理系统的吞吐量？

**优化策略**：

**1. 并发调优**

```python
# 找到最优的 max_workers
# 太小：GPU 利用率低
# 太大：排队严重，延迟增加
for max_workers in [10, 20, 30, 50]:
    throughput = benchmark(max_workers)
    print(f"max_workers={max_workers}: throughput={throughput} questions/hour")
```

**2. Prefix Caching**

```python
# 同一问题的多次 rollout 共享 system prompt
# vLLM 支持 prefix caching，但需要同一端口
# 这就是 sticky assignment 的意义
```

**3. 批处理优化**

```python
# 当前实现是每个 Agent 独立调用 LLM
# 可以优化：将多个 Agent 的 LLM 调用合并为一个 batch
# 但这需要修改 ReAct 循环的实现
```

**4. 工具调用并行化**

```python
# 当前实现是串行调用工具
# 可以优化：当模型输出多个独立的 tool_call 时，并行执行
# 但这需要修改 tool_call 的解析逻辑
```

---

## 第五类：业务与场景理解

> 核心要求：项目真正需要产生**场景价值和业务价值**，关注上线成本和优化优先级

---

### Q5.1: Deep Research Agent 适合什么样的业务场景？

**适合的场景**：

| 场景 | 特点 | 价值 |
|------|------|------|
| **学术研究** | 需要多源检索、高准确性 | 节省研究者 80% 的文献调研时间 |
| **市场调研** | 需要综合多个来源的信息 | 快速生成市场分析报告 |
| **竞品分析** | 需要跨平台收集信息 | 自动化竞品监控 |
| **法律合规** | 需要检索法规和案例 | 降低合规风险 |
| **技术支持** | 需要查阅文档和知识库 | 提高问题解决效率 |

**不适合的场景**：

| 场景 | 原因 |
|------|------|
| **简单问答** | 延迟太高（分钟级），不如直接用 ChatGPT |
| **实时对话** | 响应时间无法满足实时交互需求 |
| **创意写作** | 不需要外部信息检索 |
| **代码生成** | 主要依赖模型能力，不需要 Agent |
| **隐私敏感** | 需要调用外部 API，存在数据泄露风险 |

**用户画像**：

```
主要用户：需要深度信息研究的专业人士
- 研究人员、分析师、咨询师
- 他们的问题通常需要：
  1. 多个信息源的综合
  2. 跨语言的信息检索
  3. 时效性要求不极端（可以等几分钟）
  4. 准确性要求高（宁可慢也不能错）
```

---

### Q5.2: 如果资源有限，应该首先优化哪些部分？

**成本分析**：

```
单次 Agent 推理的成本分解：
├── LLM 推理: ~$0.1-0.5 (取决于 token 数)
├── Serper API: ~$0.001/查询
├── Jina API: ~$0.001/页面
├── SandboxFusion: ~$0.01/执行
└── 总计: ~$0.1-0.5/问题

假设每天处理 1000 个问题：
- 日成本: $100-500
- 月成本: $3000-15000
```

**优化优先级排序**：

| 优先级 | 优化方向 | 预期收益 | 实现难度 |
|--------|---------|---------|---------|
| **P0** | 减少无效工具调用 | 降低成本 30-50% | 中 |
| **P0** | 提高 Pass@1 | 减少重复推理 | 高 |
| **P1** | 引入缓存机制 | 降低重复查询成本 | 低 |
| **P1** | 优化摘要模型 | 降低 Visit 延迟 | 中 |
| **P2** | 引入主动终止 | 减少无效推理轮次 | 中 |
| **P2** | 模型量化 | 降低 GPU 成本 | 低 |

**P0 优化：减少无效工具调用**

```python
# 当前问题：模型可能反复调用相同的搜索查询
# 解决方案：缓存已执行的工具调用
tool_cache = {}

def call_with_cache(tool_name, tool_args):
    cache_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
    if cache_key in tool_cache:
        return tool_cache[cache_key]
    result = call_tool(tool_name, tool_args)
    tool_cache[cache_key] = result
    return result
```

**P0 优化：提高 Pass@1**

```python
# 当前 Pass@3 = 60% 意味着有 40% 的问题无论如何都答不对
# 优化方向：
# 1. 分析失败案例的模式
# 2. 改进 system prompt
# 3. 增加更多工具（如计算器、翻译器）
# 4. 改进数据合成质量
```

---

### Q5.3: 如何评估 Agent 系统的 ROI？

**成本计算**：

```python
monthly_costs = {
    "gpu_rental": 5000,      # 8×A100 月租
    "api_costs": 3000,       # Serper + Jina + 其他
    "engineering": 20000,    # 工程师人力
    "total": 28000,
}
```

**价值计算**：

```python
# 方法1: 时间节省价值
user_time_saved_per_query = 30  # 分钟
queries_per_month = 10000
hourly_rate = 100  # 研究人员时薪
time_value = (user_time_saved_per_query / 60) * queries_per_month * hourly_rate
# = $500,000/月

# 方法2: 任务完成率提升
baseline_completion_rate = 0.3  # 没有 Agent 时的任务完成率
agent_completion_rate = 0.7     # 有 Agent 时的任务完成率
value_per_completed_task = 50   # 每个完成任务的价值
monthly_value = (agent_completion_rate - baseline_completion_rate) * queries_per_month * value_per_completed_task
# = $200,000/月
```

**ROI 计算**：

```python
roi = (monthly_value - monthly_costs["total"]) / monthly_costs["total"]
# = (200000 - 28000) / 28000 = 614%
```

---

### Q5.4: 如何设计 Agent 系统的灰度发布策略？

**灰度阶段设计**：

```
阶段1: 内部测试 (1周)
├── 用户: 内部团队 (10人)
├── 流量: 100 queries/day
├── 目标: 验证功能正确性
└── 指标: 错误率 < 5%

阶段2: 小范围灰度 (2周)
├── 用户: 5% 外部用户
├── 流量: 1000 queries/day
├── 目标: 验证稳定性
└── 指标: Pass@1 > 50%, 延迟 < 5min

阶段3: 大范围灰度 (2周)
├── 用户: 50% 外部用户
├── 流量: 10000 queries/day
├── 目标: 验证规模化
└── 指标: 成本可控, 用户满意度 > 4.0

阶段4: 全量发布
├── 用户: 100% 用户
├── 流量: 无限制
└── 持续监控
```

**灰度发布的技术实现**：

```python
# 基于用户 ID 的分流
def get_model_version(user_id):
    hash_value = hash(user_id) % 100
    if hash_value < 5:      # 5% 灰度
        return "v2"
    else:                    # 95% 旧版本
        return "v1"

# 基于问题类型的分流
def get_model_version_by_type(question_type):
    if question_type in ["simple_qa", "factual"]:
        return "v1"  # 简单问题用旧模型
    else:
        return "v2"  # 复杂问题用新模型
```

---

### Q5.5: 如果要将 DeepResearch 部署为 SaaS 服务，需要考虑哪些工程问题？

**关键工程问题**：

**1. 多租户隔离**

```python
# 每个租户的 API key、配额、数据隔离
tenant_config = {
    "tenant_id": "company_a",
    "api_key": "sk-xxx",
    "monthly_quota": 10000,
    "data_retention": "30d",
    "model_version": "v2",
}
```

**2. 计费系统**

```python
# 按 token 计费
billing = {
    "input_token_price": 0.001,   # $/1K tokens
    "output_token_price": 0.002,  # $/1K tokens
    "tool_call_price": 0.01,      # $/call
    "minimum_charge": 0.01,       # 最低收费
}
```

**3. 限流和配额**

```python
# 令牌桶限流
rate_limiter = TokenBucket(
    rate=10,      # 10 requests/second
    capacity=50,  # burst up to 50
)

# 月度配额
monthly_quota = {
    "free": 100,
    "pro": 10000,
    "enterprise": 100000,
}
```

**4. 数据安全**

```python
# 用户数据加密存储
# API 调用不保留用户数据
# 定期清理临时文件
# 合规审计日志
```

**5. 高可用设计**

```
┌─────────────────────────────────────────────┐
│              负载均衡层 (Nginx/ALB)           │
├─────────────────────────────────────────────┤
│           API Gateway (认证/限流)             │
├─────────────────────────────────────────────┤
│          Agent 调度层 (K8s Pod)              │
├─────────────────────────────────────────────┤
│         vLLM Serving 层 (GPU Pool)           │
├─────────────────────────────────────────────┤
│         工具服务层 (Search/Visit/Python)      │
└─────────────────────────────────────────────┘
```

---

## 附录：面试回答技巧总结

### 回答框架：STAR + 深度

| 步骤 | 内容 | 示例 |
|------|------|------|
| **S**ituation | 背景和问题 | "Agent 推理需要处理超长上下文..." |
| **T**ask | 你的职责 | "我需要设计一个高效的推理系统..." |
| **A**ction | 具体行动 | "我分析了代码，发现主要瓶颈在..." |
| **R**esult | 结果和数据 | "优化后延迟降低了 40%..." |
| **+Depth** | 底层原理 | "根本原因是 Attention 的 O(n²) 复杂度..." |
| **+Limitation** | 局限性 | "但这个方案在 X 场景下仍有问题..." |
| **+Improvement** | 改进方向 | "未来可以考虑引入稀疏注意力..." |

### 面试中的追问应对

| 追问类型 | 应对策略 |
|---------|---------|
| "为什么这样设计？" | 从问题出发，说明设计决策的 trade-off |
| "有没有更好的方案？" | 承认局限性，提出 2-3 个替代方案 |
| "实际效果如何？" | 给出具体数据，包括对比实验 |
| "遇到了什么困难？" | 描述问题 → 分析原因 → 解决方案 |
| "如果重新做会怎么改？" | 基于经验教训的改进思路 |

---

*最后更新: 2026-06-27*
*基于 Tongyi DeepResearch 项目代码分析生成*
*针对 LLM & Agent 算法实习面试*
