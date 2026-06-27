# LLM & Agent 算法实习面试准备指南

> 基于 Tongyi DeepResearch 项目深度分析，面向 LLM Agent 应用与后训练方向的面试准备文档

---

## 目录

- [第一部分：项目全景概述](#第一部分项目全景概述)
- [第二部分：Agent 推理架构深度剖析](#第二部分agent-推理架构深度剖析)
- [第三部分：训练流水线全链路解析](#第三部分训练流水线全链路解析)
- [第四部分：数据合成方法论](#第四部分数据合成方法论)
- [第五部分：强化学习在 Agent 训练中的应用](#第五部分强化学习在-agent-训练中的应用)
- [第六部分：工程系统设计要点](#第六部分工程系统设计要点)
- [第七部分：高频面试问题与深度回答](#第七部分高频面试问题与深度回答)
- [第八部分：面试者自我介绍框架](#第八部分面试者自我介绍框架)
- [第九部分：延伸阅读与论文清单](#第九部分延伸阅读与论文清单)

---

## 第一部分：项目全景概述

### 1.1 项目定位

Tongyi DeepResearch 是通义实验室开发的**深度信息检索 Agent 模型**，核心特点：

- **30.5B 总参数，每 token 仅激活 3.3B**（MoE 架构，基于 Qwen3-30B-A3B）
- 专注于**长周期、深度信息检索任务**（long-horizon, deep information-seeking）
- 在 BrowseComp、GAIA、HLE、WebWalkerQA 等多个 Agentic Benchmark 上达到 SOTA

### 1.2 技术栈速览

```
┌──────────────────────────────────────────────────────────┐
│                    Tongyi DeepResearch                    │
├──────────────────────────────────────────────────────────┤
│  推理层: ReAct Agent + vLLM Serving + Multi-GPU 并行      │
│  工具层: Search(Serper) + Visit(Jina) + Scholar + Python  │
│  训练层: Agentic CPT → SFT → RL(GRPO变体)                │
│  数据层: 自动化数据合成 pipeline (FAS/HAS/SailorFog-QA)   │
│  评估层: BrowseComp / GAIA / HLE / WebWalkerQA / FRAMES  │
└──────────────────────────────────────────────────────────┘
```

### 1.3 Deep Research Agent 家族演进

| 项目 | 核心贡献 | 关键词 |
|------|---------|--------|
| WebWalker | Web 遍历 Benchmark | 评测基准 |
| WebDancer | 端到端 Agentic 训练框架 | 四阶段训练范式 |
| WebSailor | 高难度不确定场景推理 | DUPO算法, SailorFog-QA |
| WebShaper | 形式化驱动的数据合成 | 知识投影(KP), R-Union |
| AgentFounder | Agentic 持续预训练 | CPT, 开放世界记忆 |
| AgentScaler | 环境扩展 | 函数调用场景扩展 |
| **DeepResearch** | 终极集成 | MoE + CPT + SFT + RL |

---

## 第二部分：Agent 推理架构深度剖析

### 2.1 ReAct 范式实现

**核心文件**: `inference/react_agent.py`

ReAct (Reasoning + Acting) 是本项目的推理核心范式，模型交替进行**思考**和**行动**：

```
用户问题 → [Think → Act → Observe] × N轮 → <answer>最终答案</answer>
```

**代码级细节**（面试可深挖点）：

```python
# react_agent.py 核心循环
while num_llm_calls_available > 0:
    content = self.call_server(messages, planning_port)  # 调用LLM
    
    if '<tool_call>' in content:  # 模型决定调用工具
        tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
        result = self.custom_call_tool(tool_name, tool_args)
        messages.append({"role": "user", "content": "<tool_response>\n" + result + "\n</tool_response>"})
    
    if '<answer>' in content:  # 模型决定给出最终答案
        break
```

**面试深挖点**：
1. **为什么 observation 以 user role 而非 system role 注入？** — 因为 OpenAI API 兼容格式中，tool result 通常作为 user 消息传递，且多轮对话中需要保持 role 一致性
2. **停止条件设计**：`\n</tool_response>` 和 `<tool_call>` 都作为 stop token，防止模型生成不完整的工具调用
3. **上下文溢出处理**：当 token 数超过 `110*1024` 时，强制模型基于已有信息给出答案，这体现了**长上下文管理策略**

### 2.2 工具系统设计

项目定义了 5 个核心工具：

| 工具 | 实现文件 | 关键设计 |
|------|---------|---------|
| `search` | `tool_search.py` | 批量查询支持，中英文自动检测，Serper API |
| `visit` | `tool_visit.py` | Jina 读取 + LLM 摘要提取，支持多 URL 并行 |
| `google_scholar` | `tool_scholar.py` | 学术搜索，返回引用数/PDF链接 |
| `PythonInterpreter` | `tool_python.py` | 沙箱执行(SandboxFusion)，负载均衡多端点 |
| `parse_file` | `tool_file.py` | PDF/DOCX/PPTX/视频等多模态文件解析 |

**面试深挖点**：

1. **Visit 工具的两阶段设计**：
   - 第一阶段：Jina Reader 获取原始网页内容
   - 第二阶段：用另一个 LLM（Summary Model）做信息提取，输出结构化 JSON（rational/evidence/summary）
   - **为什么需要两阶段？** — 原始网页内容太长（可达 95K tokens），需要基于用户 goal 做定向信息抽取

2. **PythonInterpreter 的容错机制**：
   ```python
   for attempt in range(8):  # 最多重试8次
       endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)  # 随机选择端点
       code_result = run_code(...)  # 负载均衡
   ```
   - 多端点随机选择实现负载均衡
   - 超时检测：`execution_time >= timeout-1`

3. **Search 工具的中英文处理**：
   - 检测中文字符 → 使用 `gl: "cn"` + `hl: "zh-cn"`
   - 英文查询 → 使用 `gl: "us"` + `hl: "en"`

### 2.3 System Prompt 设计

```python
SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct 
thorough, multi-source investigations into any topic..."""
```

关键要素：
- 工具定义采用 **JSON Schema** 格式内嵌在 XML `<tools>` 标签中
- 工具调用格式：`<tool_call>{"name": ..., "arguments": ...}</tool_call>`
- 日期感知：`Current date: {today_date}`

**面试深挖点**：为什么不用 function calling 格式而用 XML 标签？— 因为自定义格式更灵活，且不依赖特定推理框架的 function calling 实现，兼容性更好。

### 2.4 多 Rollout 推理

```python
# run_multi_react.py
for rollout_idx in range(1, roll_out_count + 1):  # 默认3次rollout
    # 每个问题分配到固定端口（sticky assignment）
    planning_port = question_to_ports[question]
```

**设计要点**：
- **Sticky Port Assignment**：同一问题的所有 rollout 分配到同一 GPU 端点，避免 KV cache 重复加载
- **数据分片**：支持 `total_splits` × `worker_split` 多机并行
- **断点续跑**：通过检查已处理的 question 集合实现

---

## 第三部分：训练流水线全链路解析

### 3.1 总体训练范式

DeepResearch 采用**三阶段训练**（参考 WebDancer 四阶段 + AgentFounder CPT）：

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Agentic CPT │ → │  SFT冷启动  │ → │  RL泛化提升  │ → │   推理部署   │
│ (AgentFounder)│   │(Trajectory) │   │ (GRPO/DUPO) │   │  (ReAct)    │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

### 3.2 阶段一：Agentic 持续预训练 (Agentic CPT)

**来源**: AgentFounder 论文

**核心创新**：将 CPT 引入 Agent 训练 pipeline，这是业界首次。

**两阶段 CPT 策略**：
1. **32K 上下文 CPT**：训练基础 Agent 能力（搜索、浏览、推理）
2. **128K 上下文 CPT**：扩展到超长上下文，处理复杂研究任务

**数据合成方法**：
- **FAS (Planning Action Synthesis)**：从多样 QA 实例生成推理-行动数据，强化**规划能力**
- **FAS (Reasoning Action Synthesis)**：结合问题和知识来源，模拟在完全信息条件下的逻辑推理过程
- **HAS (Decision-Making Action Synthesis)**：将 agent 轨迹重新构建为**多步决策过程**，在每一步探索推理-行动空间

**面试深挖点**：
1. **CPT vs SFT 的区别？** — CPT 在预训练阶段注入 Agent 能力，保持模型通用性；SFT 是任务特定的微调
2. **为什么要分 32K 和 128K？** — 训练效率：先用短上下文快速学习基础能力，再用长上下文适应复杂场景；同时 128K 训练的计算成本远高于 32K
3. **开放世界记忆 (Open-World Memory)**：将持续更新的数据流转化为开放世界记忆，用于合成多样 QA 格式

### 3.3 阶段二：监督微调 (SFT)

**来源**: WebDancer 训练范式

**四步流程**：
1. **Browsing Data Construction**：从网页构建 QA 数据
2. **Trajectory Sampling**：采样成功的 agent 交互轨迹
3. **SFT 冷启动**：用高质量轨迹做监督微调
4. **RL 泛化**：进一步通过 RL 提升泛化能力

**SFT 数据格式**（ReAct 格式）：
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
    {"role": "user", "content": "<tool_response>..."},
    {"role": "assistant", "content": "...<answer>最终答案</answer>"}
  ]
}
```

**面试深挖点**：
1. **为什么需要 trajectory-level 而非 turn-level 的 SFT？** — Agent 任务的成功依赖于多步决策的组合，单步正确不代表整体正确；trajectory-level 保证了全局一致性
2. **冷启动的重要性**：RL 从零开始探索效率极低，SFT 提供了一个合理的初始化策略，大幅降低 RL 的搜索空间
3. **推理重建 (Reasoning Reconstruction)**：WebSailor 中提出从专家轨迹中重建简洁的推理过程，避免 teacher model 的风格和冗长问题

### 3.4 阶段三：强化学习 (RL)

**来源**: WebSailor DUPO + DeepResearch GRPO 变体

**DeepResearch 的 RL 特点**（来自 README）：
- 严格 **on-policy** RL
- 基于定制的 **Group Relative Policy Optimization (GRPO)** 框架
- **Token-level 策略梯度**（而非 trajectory-level）
- **Leave-one-out 优势估计**
- **负样本选择性过滤**以稳定非平稳环境中的训练

**DUPO (Duplicating Sampling Policy Optimization)**：
- WebSailor 提出的高效 Agentic RL 算法
- 在 RFT 冷启动后进一步优化

---

## 第四部分：数据合成方法论

### 4.1 WebShaper：形式化驱动的数据合成

**核心思想**：用**知识投影 (Knowledge Projection, KP)** 形式化信息检索任务。

**关键概念**：
- **知识投影 (KP)**：一组实体的集合，支持两种核心操作
  - **R-Union**：从多个网页中联合提取并汇总信息
  - **Intersection**：取交集操作
- **变量/常量/目标变量**：KP 表示中的三种元素
- **Layer-wise Structure**：逐层遍历叶常量并替换为变量

**面试深挖点**：
1. **为什么形式化比 ad-hoc 方法好？** — 形式化保证了数据的多样性和系统性覆盖，避免了推理捷径（reasoning shortcuts）
2. **R-Union 的意义**：迫使模型从多个分布式网页中综合信息，模拟真实的研究过程
3. **与传统 QA 数据的区别**：传统方法先检索再组织信息再合成；WebShaper 先建立形式化框架，再收集信息合成

### 4.2 SailorFog-QA：不确定性场景数据合成

**核心创新**：生成具有**高度不确定性和困难度**的 QA 数据。

**三步流程**：
1. **知识图谱构建**：构建复杂知识图谱
2. **信息模糊化 (Information Obfuscation)**：对图谱中的信息进行模糊化处理
3. **QA 生成**：生成需要创造性探索的 QA 对

**难度分级**：
- Level 1：简单检索
- Level 2：多步检索
- Level 3：高不确定性 + 非线性推理路径（SailorFog-QA 重点）

### 4.3 HAS：高阶动作合成

**核心思想**：将 agent 轨迹重新构建为多步决策过程。

**具体做法**：
- 在轨迹的每一步，探索完整的推理-行动空间
- 扩展 agent 对动作-答案空间的探索能力
- 增强决策能力

---

## 第五部分：强化学习在 Agent 训练中的应用

### 5.1 GRPO (Group Relative Policy Optimization)

**与 PPO 的关键区别**：

| 特性 | PPO | GRPO |
|------|-----|------|
| 价值网络 | 需要 Critic Network | 不需要，用组内相对优势 |
| 优势估计 | GAE | Group Relative / Leave-one-out |
| 内存开销 | 大（Policy + Critic） | 小（仅 Policy） |
| 训练稳定性 | 需要精细调参 | 更稳定（相对比较） |

**GRPO 核心公式**（面试需掌握）：
```
对每个 prompt x:
  采样 G 个 responses: {y_1, ..., y_G}
  计算 rewards: {r_1, ..., r_G}
  优势估计: A_i = (r_i - mean(r)) / std(r)  # 组内归一化
  策略梯度: ∇J = E[∇log π(y|x) · A · clip(ratio, 1-ε, 1+ε)]
```

**DeepResearch 的 GRPO 变体特化**：
1. **Token-level 策略梯度**：不只在 sequence 末端计算 reward，而是在 token 级别传播梯度
2. **Leave-one-out 优势估计**：每个样本的优势用其余所有样本的平均 reward 作为 baseline
3. **负样本选择性过滤**：过滤掉所有 rollout 都失败的 prompt（避免无效训练信号）

### 5.2 Agent RL 的特殊挑战

**面试深挖点**：

1. **非平稳环境问题**：
   - Agent 调用的工具（搜索 API、网页内容）会随时间变化
   - 同一 action 在不同时间可能得到不同 observation
   - 解决方案：on-policy 训练 + 负样本过滤

2. **稀疏奖励问题**：
   - Agent 任务的 reward 通常只在最终答案处给出（0/1 reward）
   - 中间步骤没有直接的监督信号
   - 解决方案：token-level 策略梯度 + 合理的 reward shaping

3. **长序列问题**：
   - Agent 轨迹可能包含数十个 step，总长度超过 100K tokens
   - RL 训练中的内存和计算挑战
   - 解决方案：MoE 架构减少激活参数 + 长上下文优化

4. **探索-利用权衡**：
   - Agent 需要在已知有效策略（利用）和尝试新方法（探索）之间平衡
   - `presence_penalty=1.1` 鼓励模型探索不同的搜索策略

### 5.3 Reward 设计

**二元 Reward**：
```python
reward = 1.0 if prediction == ground_truth else 0.0
```

**Pass@K 评估**：
- 在 K 次 rollout 中至少有一次答对
- DeepResearch 默认 `roll_out_count=3`

---

## 第六部分：工程系统设计要点

### 6.1 推理系统架构

```
┌────────────────────────────────────────────────────────┐
│                    推理调度层                            │
│  run_multi_react.py (ThreadPoolExecutor, 多线程并发)    │
├────────────────────────────────────────────────────────┤
│                    Agent 层                             │
│  MultiTurnReactAgent (ReAct 循环, 工具调度)              │
├────────────────────────────────────────────────────────┤
│                    模型服务层                            │
│  vLLM × 8 GPUs (6001-6008端口, 负载均衡)                │
├────────────────────────────────────────────────────────┤
│                    工具层                               │
│  Serper API / Jina API / SandboxFusion / DashScope     │
└────────────────────────────────────────────────────────┘
```

### 6.2 关键工程设计

**1. 多 GPU 并行 serving**：
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --port 6001 &
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH --port 6002 &
# ... 8 个 GPU 各自独立 serving
```
- 每个 GPU 运行独立的 vLLM 实例
- Agent 级别的负载均衡（非 token 级别）

**2. 请求重试机制**：
```python
for attempt in range(max_tries):
    sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)  # 指数退避 + 随机抖动
    sleep_time = min(sleep_time, 30)  # 最大等待30秒
```

**3. Token 计数管理**：
```python
max_tokens = 110 * 1024  # 110K tokens 上限
token_count = self.count_tokens(messages)
if token_count > max_tokens:
    # 强制模型给出答案
```

**4. 超时控制**：
- 单次 Agent 推理：150 分钟超时
- 工具调用：各工具有独立超时设置

### 6.3 vLLM 部署优化

**面试深挖点**：
1. **为什么用 vLLM 而非 TGI/sglang？** — vLLM 的 PagedAttention 机制对长序列友好，且支持连续批处理
2. **为什么每个 GPU 独立 serving 而非 tensor parallel？** — Agent 推理的请求量不均匀，独立 serving 更灵活；且避免了 TP 的通信开销
3. **KV Cache 管理**：`--disable-log-requests` 减少日志开销，提升吞吐

---

## 第七部分：高频面试问题与深度回答

### 7.1 基础概念类

**Q1: 解释 ReAct 范式的工作原理**

> ReAct 将推理（Reasoning）和行动（Acting）交替进行。模型首先用 `<think>` 标签进行思考分析，然后决定是调用工具（`<tool_call>`）还是给出最终答案（`<answer>`）。每次工具调用后，结果以 `<tool_call>` 标签返回给模型作为新的 observation。这种交替循环使得模型能够基于外部信息迭代地修正推理。

**Q2: Agent 训练和普通 SFT/RLHF 有什么区别？**

> 核心区别有三点：
> 1. **数据结构不同**：Agent 训练的样本是多轮交互轨迹（包含 tool call 和 observation），而非单轮对话
> 2. **环境交互**：Agent RL 需要与真实环境（搜索引擎、网页）交互获取 observation，是 online RL；而 RLHF 通常基于静态偏好数据
> 3. **奖励稀疏性**：Agent 任务通常只有最终答案的 0/1 reward，中间步骤没有直接监督

**Q3: 什么是 MoE 架构？DeepResearch 为什么选择 MoE？**

> MoE (Mixture of Experts) 通过稀疏激活实现大参数量低计算成本。DeepResearch 有 30.5B 总参数但每 token 只激活 3.3B。选择 MoE 的原因：
> 1. Agent 推理需要处理超长上下文（>100K tokens），MoE 降低了每 token 的计算量
> 2. 保持大模型容量的同时控制推理延迟
> 3. 不同的 expert 可能自然分工处理不同类型的推理（搜索、数学、语言等）

### 7.2 技术深挖类

**Q4: GRPO 相比 PPO 在 Agent 训练中的优势是什么？**

> 1. **无需 Critic Network**：Agent 任务的轨迹很长（100K+ tokens），训练一个准确的 Critic 非常困难且内存开销大。GRPO 用组内相对优势代替，避免了这个问题
> 2. **更稳定的优势估计**：leave-one-out 估计天然具有 baseline，减少了方差
> 3. **适配非平稳环境**：Agent 环境中，同一 action 的结果会变化，GRPO 的 group relative 机制对此更鲁棒

**Q5: Token-level 策略梯度 vs Trajectory-level 策略梯度的区别？**

> - **Trajectory-level**：只在整个轨迹结束后计算一次 reward，所有 token 共享同一个 advantage。问题是：无法区分哪些 token 贡献了成功/失败
> - **Token-level**：在每个 token 位置计算策略梯度，允许模型在更细粒度上学习。对于 Agent 任务，这意味着模型能学到"在哪个 step 做出了关键决策"
> - 实现挑战：需要更复杂的 credit assignment 机制

**Q6: 数据合成中的 reasoning shortcuts 问题是什么？如何解决？**

> Reasoning shortcuts 指模型在合成数据中找到了捷径，不需要完整推理就能得到答案。例如：
> - 问题结构过于线性，常量直接连接到目标变量
> - 信息冗余，只需部分信息就能回答
>
> WebShaper 的解决方案：
> 1. **形式化约束**：通过 KP 形式化确保没有常量直接连接到目标变量
> 2. **Layer-wise 结构**：逐层遍历变量，迫使模型经过所有中间步骤
> 3. **R-Union 操作**：迫使模型从多个来源综合信息

**Q7: 如何处理 Agent 推理中的上下文溢出？**

> DeepResearch 的策略（`react_agent.py:186-209`）：
> 1. 持续监控 token 数量
> 2. 当超过 110K tokens 时，注入特殊提示：要求模型停止工具调用，基于已有信息给出答案
> 3. 模型被迫在有限信息下做出最佳判断
>
> 这是一种**被动截断策略**。更先进的方法（如 ReSum 论文）提出主动对历史上下文进行摘要压缩。

### 7.3 系统设计类

**Q8: 如果让你设计一个 Deep Research Agent 的训练 pipeline，你会怎么做？**

> 参考 DeepResearch 的实践，我会设计如下 pipeline：
>
> **Phase 1: 数据准备**
> - 使用 WebShaper 的形式化方法合成多样化的 QA 数据
> - 使用 SailorFog-QA 方法生成高不确定性场景数据
> - 保证数据覆盖不同难度级别和领域
>
> **Phase 2: Agentic CPT**（可选，资源允许时）
> - 在 32K 上下文上做第一阶段 CPT
> - 在 128K 上下文上做第二阶段 CPT
>
> **Phase 3: SFT 冷启动**
> - 采样成功的 agent 轨迹
> - 清洗和重建推理过程
> - 做 trajectory-level SFT
>
> **Phase 4: RL 泛化**
> - 使用 GRPO 或 DUPO 算法
> - 设计合适的 reward（准确性 + 格式正确性）
> - 过滤负样本稳定训练
>
> **Phase 5: 评估与迭代**
> - 在多个 benchmark 上评估
> - 分析失败案例，迭代数据合成策略

**Q9: 如何评估一个 Deep Research Agent 的效果？**

> 评估维度：
> 1. **准确性**：Pass@1, Pass@3（多次 rollout 的成功率）
> 2. **效率**：平均工具调用次数、平均推理时间
> 3. **鲁棒性**：在不同难度级别的泛化能力
> 4. **工具使用质量**：搜索查询质量、网页信息提取准确性
>
> 常用 Benchmark：
> - **BrowseComp**：需要深度浏览的复杂问题
> - **GAIA**：通用 AI 助手评估
> - **HLE (Humanity's Last Exam)**：高难度学术问题
> - **WebWalkerQA**：网页遍历能力
> - **FRAMES**：事实检索与多源推理

### 7.4 开放讨论类

**Q10: 当前 Deep Research Agent 的主要局限是什么？未来方向？**

> **局限**：
> 1. **推理效率**：单次推理可能需要 100+ LLM calls，延迟高
> 2. **工具依赖**：依赖外部 API（搜索、网页访问），可用性和延迟不可控
> 3. **幻觉问题**：在信息不足时可能生成错误答案
> 4. **非平稳环境**：网页内容变化导致相同查询可能得到不同结果
>
> **未来方向**：
> 1. **Test-time Scaling**：Heavy Mode / IterResearch 等推理时扩展策略
> 2. **主动上下文管理**：如 AgentFold 的主动 context 管理
> 3. **并行思考**：如 ParallelMuse 的并行推理
> 4. **置信度引导**：如 BrowseConf 的置信度评估
> 5. **多模态扩展**：如 WebWatcher 的视觉-语言 Agent

---

## 第八部分：面试者自我介绍框架

### 8.1 一分钟版本

> 我是 [姓名]，[学校] [专业] 硕士在读。我的研究方向是 LLM Agent，特别是深度信息检索 Agent 的训练与推理优化。
>
> 我系统学习了 Tongyi DeepResearch 项目的技术栈，包括 ReAct 推理范式、自动化数据合成（WebShaper 形式化方法、SailorFog-QA 不确定性数据生成）、以及 Agent 后训练全流程（Agentic CPT → SFT → GRPO 强化学习）。
>
> 在工程实践方面，我了解 vLLM 部署优化、多 GPU 并行 serving、工具系统设计等工程细节。
>
> 我对 Agent RL 中的非平稳环境问题、稀疏奖励的 credit assignment、以及 test-time scaling 等前沿问题有深入思考。

### 8.2 详细介绍框架（3-5 分钟）

**第一段：背景与动机**（30秒）
- 学校、专业、研究方向
- 为什么对 LLM Agent 感兴趣

**第二段：技术理解**（1-2分钟）
- 对 DeepResearch 项目的理解
- 能复述训练 pipeline 的核心设计决策
- 能解释关键技术（GRPO、数据合成、ReAct）

**第三段：深入分析能力**（1-2分钟）
- 对某个技术点的深入分析（如 Agent RL 的挑战）
- 提出自己的见解或改进想法

**第四段：实践与规划**（30秒）
- 相关实践经验（如有）
- 希望在实习中深入的方向

---

## 第九部分：延伸阅读与论文清单

### 9.1 核心论文（必读）

| # | 论文 | 关键贡献 |
|---|------|---------|
| 1 | [WebWalker](https://arxiv.org/pdf/2501.07572) | Web 遍历 Benchmark (ACL 2025) |
| 2 | [WebDancer](https://arxiv.org/pdf/2505.22648) | 四阶段训练范式 (NeurIPS 2025) |
| 3 | [WebSailor](https://arxiv.org/pdf/2507.02592) | DUPO 算法 + SailorFog-QA |
| 4 | [WebShaper](https://arxiv.org/pdf/2507.15061) | 形式化驱动的数据合成 |
| 5 | [AgentFounder](https://arxiv.org/pdf/2509.13310) | Agentic 持续预训练 |
| 6 | [DeepResearch Tech Report](https://arxiv.org/pdf/2510.24701) | 综合技术报告 |

### 9.2 扩展论文

| # | 论文 | 关键词 |
|---|------|--------|
| 7 | WebWatcher | 多模态 Agent |
| 8 | ReSum | 上下文摘要 |
| 9 | ParallelMuse | 并行思考 |
| 10 | AgentFold | 主动上下文管理 |
| 11 | WebLeaper | 高效信息检索 |
| 12 | BrowseConf | 置信度引导 |
| 13 | AgentFrontier | ZPD 引导数据合成 |

### 9.3 基础知识论文

- **ReAct**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **GRPO**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
- **DAPO**: [DAPO: An Open-Source LLM Reinforcement Learning System](https://arxiv.org/abs/2503.14476)
- **vLLM**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### 9.4 面试前 Checklist

- [ ] 能画出 DeepResearch 的整体架构图
- [ ] 能解释 ReAct 循环的完整流程（包括 stop token 设计）
- [ ] 能描述 GRPO 与 PPO 的区别和公式
- [ ] 能说明数据合成的三种方法（FAS/HAS/SailorFog-QA）
- [ ] 能讨论 Agent RL 的主要挑战和解决方案
- [ ] 能解释 MoE 架构在 Agent 场景的优势
- [ ] 能描述工程系统的多 GPU serving 方案
- [ ] 准备了至少 2 个"你有什么问题想问我们"的问题

---

## 附录：代码关键路径速查

| 功能 | 文件路径 | 关键行号 |
|------|---------|---------|
| ReAct 主循环 | `inference/react_agent.py` | L120-226 |
| System Prompt | `inference/prompt.py` | L1-35 |
| 信息提取 Prompt | `inference/prompt.py` | L37-51 |
| 搜索工具 | `inference/tool_search.py` | L18-131 |
| 网页访问工具 | `inference/tool_visit.py` | L39-256 |
| Python 沙箱 | `inference/tool_python.py` | L28-150 |
| 学术搜索 | `inference/tool_scholar.py` | L13-110 |
| 多 Rollout 调度 | `inference/run_multi_react.py` | L13-229 |
| 推理启动脚本 | `inference/run_react_infer.sh` | L1-117 |
| 评估脚本 | `evaluation/evaluate_deepsearch_official.py` | - |
| 环境配置 | `.env.example` | L1-98 |

---

*最后更新: 2026-06-27*
*基于 Tongyi DeepResearch 项目代码分析生成*
