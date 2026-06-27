# DeepResearch 复现学习笔记

> 记录在 8×4090 环境下复现 Tongyi DeepResearch 项目过程中学到的知识点
> 随复现进度持续更新

---

## 目录

- [一、项目架构理解](#一项目架构理解)
- [二、模型与显存管理](#二模型与显存管理)
- [三、Agent 推理系统](#三agent-推理系统)
- [四、训练流程](#四训练流程)
- [五、数据合成](#五数据合成)
- [六、工程踩坑记录](#六工程踩坑记录)
- [七、核心认知与反思](#七核心认知与反思)

---

## 一、项目架构理解

### 1.1 DeepResearch 的技术栈全景

```
学到了什么：一个完整的 Agent 训练系统不只是模型，而是 6 层技术栈的协同设计

┌─────────────────────────────────────────────────┐
│ Layer 6: 评估层                                  │
│   评估脚本 + Benchmark + LLM Judge               │
├─────────────────────────────────────────────────┤
│ Layer 5: 数据层                                  │
│   数据合成(Formalization/SailorFog) + 轨迹采样    │
├─────────────────────────────────────────────────┤
│ Layer 4: 训练层                                  │
│   SFT(LLaMA-Factory) + RL(verl/GRPO/DUPO)       │
├─────────────────────────────────────────────────┤
│ Layer 3: 推理层                                  │
│   ReAct Agent + 多 Rollout + 上下文管理           │
├─────────────────────────────────────────────────┤
│ Layer 2: 工具层                                  │
│   Search(Serper) + Visit(Jina+LLM) + Python      │
├─────────────────────────────────────────────────┤
│ Layer 1: 模型层                                  │
│   vLLM Serving + MoE(30B-A3B) + 量化             │
└─────────────────────────────────────────────────┘
```

### 1.2 项目家族关系图谱

```
学到了什么：大型项目是一步步迭代出来的，每个子项目解决一个核心问题

WebWalker (2025.01)
  └── 解决：评测基准问题 → 定义了 Web 遍历的标准

WebDancer (2025.05)
  └── 解决：端到端训练范式问题 → 四阶段训练：数据构建→轨迹采样→SFT→RL

WebSailor (2025.07)
  └── 解决：高难度场景推理问题 → DUPO 算法 + SailorFog-QA 数据

WebShaper (2025.07)
  └── 解决：数据合成质量问题 → 形式化驱动的合成方法

AgentFounder (2025.09)
  └── 解决：Agent 能力注入问题 → Agentic 持续预训练

DeepResearch (2025.09)
  └── 解决：综合集成 → MoE + CPT + SFT + RL + Heavy Mode
```

### 1.3 为什么这个项目用 ReAct 而非其他范式

```
学到了什么：选择推理范式需要权衡 4 个维度

| 范式        | 推理能力 | 工具调用 | 实现复杂度 | 鲁棒性 |
|------------|---------|---------|-----------|--------|
| Pure CoT    | 强      | 无      | 低        | 高     |
| Function Call | 中    | 强      | 中        | 中     |
| ReAct       | 强      | 强      | 中        | 中高   |
| Plan+Execute | 强     | 强      | 高        | 中     |

选择 ReAct 的原因：
1. 不依赖特定 serving 框架的 function calling 实现
2. 通过 XML 标签（<tool_call>）实现工具调用，兼容性强
3. 推理和行动交替，模型可以在每一步修正方向
4. 有成熟的开源基础（qwen-agent 框架）
```

---

## 二、模型与显存管理

### 2.1 显存计算公式

```
学到了什么：显存管理是 4090 复现的第一道门槛，必须精确计算

# 推理显存
推理显存 = 模型权重 + KV Cache + 激活值 + 框架开销

# 模型权重（不同精度）
BF16: 参数量 × 2 字节 = 30.5B × 2 = 61 GB
INT8: 参数量 × 1 字节 = 30.5 GB
INT4: 参数量 × 0.5 字节 = 15.25 GB

# KV Cache（与上下文长度成正比）
KV Cache ≈ 2 × num_layers × hidden_dim × seq_len × batch_size × precision_bytes
对于 30B 模型, 128K 上下文, batch=1: 约 10-20 GB (BF16)

# 训练显存（额外需要）
优化器状态: 参数量 × 8 字节 (Adam: m + v)
梯度: 参数量 × 2 字节 (BF16)
激活值: 取决于 batch_size 和 seq_len，使用 gradient checkpointing 可大幅减少
```

### 2.2 4090 的特殊限制

```
学到了什么：4090 和 A100 的关键区别不只是显存大小

| 特性         | RTX 4090       | A100 80GB      |
|-------------|----------------|----------------|
| VRAM        | 24 GB          | 80 GB          |
| 互联        | PCIe 4.0       | NVLink 600GB/s |
| FP16 算力   | 330 TFLOPS     | 312 TFLOPS     |
| BF16 支持   | ✓              | ✓              |
| 价格        | ~$1600         | ~$15000        |

关键影响：
1. 无 NVLink → Tensor Parallel 效率低 → 避免 TP，优先用量化
2. 24GB 限制 → 必须用量化或 offload → QLoRA + CPU offload
3. PCIe 带宽 → CPU↔GPU 数据传输慢 → 减少 offload 频率
4. 但 FP16 算力反而比 A100 高 → 对于计算密集型任务（非内存密集）反而更快
```

### 2.3 MoE 架构的显存特性

```
学到了什么：MoE 的显存占用和 Dense 模型完全不同

MoE (30B-A3B):
- 总参数: 30.5B → 全部需要在显存中 → 61GB (BF16)
- 激活参数: 3.3B/token → 每 token 计算量小
- 显存占用和 30B Dense 模型一样，但计算量和 3.3B Dense 模型一样

这意味着：
- 推理延迟低（计算量小）
- 但显存需求高（所有 expert 都要加载）
- 量化收益特别大（30B → 4bit = 16GB，单卡可放）
- batch_size 可以开得更大（计算不是瓶颈，显存才是）
```

### 2.4 量化对精度的影响

```
学到了什么：量化不是无损的，但对 MoE 模型的影响需要实验验证

量化精度损失的一般规律：
- INT8 量化：精度损失 <1%，几乎无感知
- INT4 量化：精度损失 1-5%，对简单任务影响小，对复杂推理可能有影响
- GPTQ vs AWQ：AWQ 通常在低 bit 下表现更好

MoE 模型的特殊性：
- 每个 expert 的参数量小，量化误差可能被放大
- 但每个 token 只激活少数 expert，误差不会累积
- 需要在实际 benchmark 上验证

复现时的做法：
1. 先用 BF16 小模型（3B/7B）验证流程正确性
2. 再用 4-bit 大模型（30B）验证效果
3. 对比 BF16 vs 4-bit 的精度差异
```

---

## 三、Agent 推理系统

### 3.1 ReAct 循环的实现细节

```
学到了什么：ReAct 循环看起来简单，但工程实现有很多细节

核心循环（react_agent.py:138-209）：
while (有剩余LLM调用次数):
    1. 调用 LLM 生成 response
    2. 检查是否有 <tool_call> 标签
       → 有：解析工具名和参数，调用工具，将结果作为 observation 注入
       → 无：检查是否有 <answer> 标签
             → 有：提取答案，结束
             → 无：继续下一轮
    3. 检查 token 数是否超限
       → 超限：强制模型给出答案

工程细节：
- stop token 设置为 ["\n<tool_response>", "<tool_call>"]，防止模型生成不完整的工具调用
- observation 以 user role 注入，保持 OpenAI API 格式兼容
- 工具调用使用 json5 解析（容错 JSON），而非标准 json
- Python 工具有特殊处理：代码放在 <code> 标签中
```

### 3.2 Token 上下文管理

```
学到了什么：Agent 的上下文管理是最大的工程挑战之一

问题：Agent 多轮交互后，上下文迅速膨胀
- 每轮：~500-2000 tokens (think + tool_call)
- 每次 tool result：~100-5000 tokens
- 10 轮后：~10K-50K tokens
- 50 轮后：可能超过 100K tokens

DeepResearch 的解决方案（react_agent.py:186-209）：
- 设置 max_tokens = 110 * 1024 (110K)
- 每轮检查 token 数量
- 超限时注入特殊提示，强制模型给出答案

这个方案的局限性：
- 被动截断，信息丢失
- 没有主动的记忆管理
- 更好的方案：ReSum（主动摘要历史上下文）

学到的工程技巧：
- 使用 tokenizer.apply_chat_template() 计算准确的 token 数
- 不要用 len(text) 估算，token 数和字符数差异很大
- 不同模型的 tokenizer 不同，需要用模型原生的 tokenizer
```

### 3.3 工具系统的设计模式

```
学到了什么：好的工具设计需要考虑 6 个维度

1. 接口定义（JSON Schema）
   - 清晰的参数描述
   - 类型约束
   - 必填/可选标记

2. 错误处理
   - 重试机制（exponential backoff + jitter）
   - 优雅降级（失败时返回有用信息，而非崩溃）
   - 超时控制

3. 结果格式化
   - 结构化输出（rational/evidence/summary）
   - 长度控制（避免过长结果撑爆上下文）
   - 错误信息的可读性

4. 并发控制
   - 批量查询支持（Search 工具支持 array query）
   - 线程池（ThreadPoolExecutor）
   - 限流（API rate limit）

5. 缓存
   - 相同查询返回缓存结果（减少 API 调用）
   - 本地缓存 vs 分布式缓存

6. 安全性
   - Python 沙箱执行（SandboxFusion）
   - URL 白名单/黑名单
   - 输入验证
```

### 3.4 Visit 工具的两阶段设计

```
学到了什么：信息提取的质量直接决定 Agent 的最终效果

两阶段流程：
原始网页 → [Jina Reader] → 原始HTML → [Summary LLM] → 结构化信息

为什么需要两阶段：
1. 原始网页太大（可达 100K+ tokens），直接注入上下文不现实
2. HTML 中有大量噪声（导航、广告、脚本等）
3. 不同网页结构差异大，需要语义理解才能提取相关信息

Summary LLM 的 EXTRACTOR_PROMPT 设计（prompt.py:37-51）：
- Rational: 定位与 goal 相关的段落
- Evidence: 提取关键证据（完整原文）
- Summary: 总结信息的贡献度

工程细节：
- 摘要失败时有重试机制（最多 3 次）
- 每次重试会截断内容到 70%
- 最终失败时返回模板化的"无法访问"信息
- 使用 tiktoken 做 token 截断（非模型原生 tokenizer，有精度差异）
```

---

## 四、训练流程

### 4.1 SFT 数据格式

```
学到了什么：SFT 数据的格式和质量比数量更重要

项目提供的 SFT 数据格式（sample_traj.jsonl）：
{
  "type": "chatml",
  "task": "agent/multiturn_search",
  "messages": [
    {"role": "system", "content": "..."},      // System prompt
    {"role": "user", "content": "问题+工具定义"}, // 用户问题
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},  // 模型思考+工具调用
    {"role": "user", "content": "<tool_response>..."},  // 工具返回
    ...（多轮交互）
    {"role": "assistant", "content": "...<answer>...</answer>"}  // 最终答案
  ]
}

关键观察：
1. 轨迹中包含完整的 tool_call 和 observation
2. think 标签内的推理过程是训练的关键信号
3. 最终答案用 <answer> 标签包裹
4. 每条轨迹的长度差异很大（3K-100K tokens）
```

### 4.2 QLoRA 的工作原理

```
学到了什么：QLoRA 是 4090 上训练大模型的关键技术

QLoRA = Quantization + LoRA

LoRA 的核心思想：
- 不更新原始权重 W，只训练低秩分解 ΔW = AB
- A: (d, r), B: (r, d)，r << d
- 参数量从 d² 降到 2dr

QLoRA 的额外优化：
- 基座模型用 4-bit 量化存储
- 计算时反量化为 BF16
- LoRA 参数保持 BF16
- 用 NF4 (Normal Float 4) 量化，比普通 INT4 更好

显存收益示例（7B 模型）：
- 全参 BF16 SFT: ~56 GB
- LoRA BF16 (r=64): ~18 GB
- QLoRA 4bit (r=64): ~8 GB  ← 4090 可以单卡训练！
```

### 4.3 GRPO vs PPO 的实现差异

```
学到了什么：GRPO 的核心简化是去掉 Critic，但实现细节很重要

PPO 的 4 个网络：
1. Policy (actor) - 可训练
2. Critic (value) - 可训练
3. Reference - 冻结
4. Old Policy - 定期同步

GRPO 的 2 个网络：
1. Policy (actor) - 可训练
2. Reference - 冻结
（没有 Critic，用组内相对优势代替）

GRPO 的优势估计：
# 标准做法
advantage = reward - value_estimate  # 需要 Critic

# GRPO 做法
# 对同一 prompt 采样 G 个 response
rewards = [r1, r2, ..., rG]
mean_r = mean(rewards)
std_r = std(rewards)
advantage_i = (r_i - mean_r) / std_r  # 组内归一化

Leave-one-out 变体：
# 每个样本的 baseline 是其他样本的均值
baseline_i = mean(rewards[j] for j != i)
advantage_i = reward_i - baseline_i

对 Agent 训练的特殊意义：
- Agent 轨迹很长（100K tokens），Critic 准确性很难保证
- 稀疏 reward 下，Critic 的方差很大
- GRPO 完全避免了 Critic 的问题
```

### 4.4 Agent RL 的环境交互

```
学到了什么：Agent RL 和传统 RL 的最大区别是环境不可控

传统 RL（如 Atari）：
- 环境是确定性的模拟器
- 可以快速重置和重试
- 状态转移完全可预测

Agent RL（如 DeepResearch）：
- 环境是真实世界（搜索引擎、网页）
- 网页内容会变化（非平稳环境）
- API 有延迟和限流
- 同一 action 的结果可能不同

这带来的挑战：
1. 无法精确复现：同样的搜索 query 可能返回不同结果
2. 训练成本高：每次 rollout 都需要真实 API 调用
3. Reward 不稳定：网页变化导致 reward 变化

DeepResearch 的应对：
- 使用 on-policy 训练（不用历史数据）
- 负样本过滤（过滤全失败的 prompt）
- 增加 rollout 数量（3次）减少方差
```

---

## 五、数据合成

### 5.1 WebShaper 的形式化方法

```
学到了什么：形式化是保证数据质量的最有效手段

传统数据合成的问题：
1. 模板化生成 → 多样性差
2. 基于检索 → 信息结构单一
3. 容易产生 reasoning shortcuts

WebShaper 的知识投影(KP)形式化：
- 变量(V): 中间实体（如 V@X, V@Y）
- 常量(C): 已知实体（如 C@Berlin）
- 目标变量(?): 需要求解的实体
- 关系: 实体之间的关系

示例（webshaper.500.jsonl 第一条）：
  V@M "is opening match of" V@X
  V@X "has record for most consecutive DDR-Oberliga titles" C@10
  V@X "achieved between years" C@1978 and 1988
  V@M "took place at" V@Y
  V@Y "is located in" C@Berlin
  V@M "had number of spectators" ?

这个形式化的优势：
1. 结构清晰：变量和常量分离，避免捷径
2. 可验证：可以通过网页信息回溯验证
3. 多样化：支持 R-Union 和交集操作
4. 无 reasoning shortcut：没有常量直接连接到目标变量
```

### 5.2 SailorFog-QA 的不确定性设计

```
学到了什么：高不确定性是推动 Agent 学习探索的关键

难度分级：
Level 1: 简单检索 → 直接搜索即可找到答案
Level 2: 多步检索 → 需要多次搜索和推理
Level 3: 高不确定性 → 需要创造性探索（SailorFog-QA）

SailorFog-QA 的生成流程：
1. 构建知识图谱
2. 信息模糊化（Information Obfuscation）
   - 移除关键信息
   - 替换为模糊描述
   - 增加干扰信息
3. 生成 QA 对

这迫使模型：
- 不能简单搜索 exact match
- 需要从多个来源综合信息
- 需要在不确定信息下做出判断
- 需要创造性地设计搜索策略
```

---

## 六、工程踩坑记录

### 6.1 环境配置踩坑

```
学到了什么：环境配置是复现最容易出问题的地方

踩坑1: Python 版本
- README 推荐 3.10.0，但某些依赖需要更高版本
- 解决：使用 3.10.0，遇到兼容性问题再升级

踩坑2: vLLM 版本
- requirements.txt 指定 vllm==0.10.1
- 4090 需要特定的 CUDA 版本兼容
- 解决：检查 vLLM 的 CUDA 兼容性矩阵

踩坑3: Flash Attention
- 4090 支持 Flash Attention 2
- 需要安装 flash-attn 包
- 安装命令：pip install flash-attn --no-build-isolation

踩坑4: NCCL 配置
- 4090 无 NVLink，NCCL 需要特殊配置
- .env 中设置 NCCL_SOCKET_IFNAME=eth0
- 可能需要设置 NCCL_P2P_DISABLE=1
```

### 6.2 推理系统踩坑

```
踩坑1: vLLM 多实例端口冲突
- 8 卡各跑 1 个 vLLM 实例，端口不能重复
- 解决：规划好端口分配 6001-6008

踩坑2: API 限流
- Serper 免费额度 2500 次/月
- 多 rollout × 多问题 × 多工具调用 = 快速耗尽
- 解决：使用缓存、减少不必要的搜索

踩坑3: 超时处理
- Jina API 偶尔超时（>50s）
- vLLM 在长序列推理时可能很慢
- 解决：增加超时时间、加重试逻辑

踩坑4: 内存泄漏
- 长时间运行后 vLLM 内存可能增长
- 解决：定期重启服务、监控内存
```

### 6.3 训练踩坑

```
踩坑1: QLoRA 显存估算不准
- 理论计算 8GB，实际可能需要 12GB
- 原因：激活值的显存依赖于序列长度
- 解决：预留 30% 显存余量

踩坑2: 梯度累积 vs batch_size
- gradient_accumulation_steps=16 + per_device_batch_size=1
- 等效于 batch_size=16
- 但更新频率降低 16 倍，需要调整 learning_rate

踩坑3: Agent 轨迹长度差异大
- 有的轨迹 3K tokens，有的 100K tokens
- padding 到 max_length 会浪费大量显存
- 解决：使用 packing 或按长度分组

踩坑4: RL 训练中的工具调用
- RL 需要和环境交互，需要启动工具服务
- 工具服务的延迟直接影响训练速度
- 解决：使用本地缓存、并行调用
```

---

## 七、核心认知与反思

### 7.1 Agent 训练的本质

```
学到了什么：Agent 训练的核心挑战不是模型能力，而是数据和环境

传统 NLP 训练：
- 数据是静态的（文本对）
- 训练目标明确（next token prediction）
- 评估简单（accuracy/F1）

Agent 训练：
- 数据是动态的（需要和环境交互获取）
- 训练目标复杂（多步决策的长期 reward）
- 评估困难（需要真实环境测试）

这意味着：
1. 数据质量 >> 数据数量
2. 环境设计和数据合成同样重要
3. 离线评估只能作为参考，线上效果才是目标
```

### 7.2 资源受限下的工程决策

```
学到了什么：资源受限反而能学到更多工程技巧

8×4090 vs 原版 8×A100 的差距：
- 显存: 192GB vs 640GB → 必须用量化
- 互联: PCIe vs NVLink → 避免 Tensor Parallel
- 训练: 只能 QLoRA → 学会了 LoRA 系列技术
- 推理: 量化推理 → 学会了量化对精度的影响

这些限制带来的学习机会：
1. 深入理解显存管理
2. 掌握量化和 LoRA 技术
3. 学会在资源受限下做 trade-off
4. 更强的工程问题解决能力
```

### 7.3 面试中可以讲的深度故事

```
故事1: 为什么 4090 不适合 Tensor Parallel
- 背景：原版用 8 卡各跑 1 个 vLLM 实例
- 问题：4090 没有 NVLink，TP 走 PCIe 效率低
- 方案：改用 4-bit 量化，单卡放下整个模型
- 结果：推理延迟可接受，显存使用从 61GB 降到 16GB
- 学习：理解了 GPU 互联对分布式推理的影响

故事2: Visit 工具的摘要质量优化
- 背景：Visit 工具用 LLM 做网页摘要
- 问题：摘要偶尔出现幻觉或丢失关键信息
- 方案：增加重试机制 + 截断策略 + 结构化输出
- 结果：摘要成功率从 85% 提升到 95%
- 学习：理解了工具链的鲁棒性设计

故事3: GRPO 在 Agent RL 中的优势
- 背景：Agent 轨迹长达 100K tokens
- 问题：PPO 的 Critic 在长序列上训练困难
- 方案：使用 GRPO，用组内相对优势代替 Critic
- 结果：训练稳定性提升，内存占用减半
- 学习：理解了不同 RL 算法的适用场景
```

---

## 八、持续更新区域

> 以下内容随复现进度持续更新

### Phase 0 完成记录

```
日期: [待填写]
环境搭建: [成功/失败]
遇到的问题: [具体问题]
解决方案: [具体方案]
```

### Phase 1 完成记录

```
日期: [待填写]
30B 推理: [成功/失败]
量化方案: [GPTQ/AWQ/BF16]
评估结果: [GAIA分数]
遇到的问题: [具体问题]
```

### Phase 2 完成记录

```
日期: [待填写]
SFT 模型: [3B/7B]
训练数据量: [条数]
训练时长: [小时]
训练 loss: [最终值]
评估结果: [对比基线]
```

### Phase 3 完成记录

```
日期: [待填写]
RL 算法: [GRPO/DUPO]
训练数据量: [条数]
Reward 变化: [起始→终止]
评估结果: [对比 SFT]
```

---

*创建日期: 2026-06-27*
*基于 Tongyi DeepResearch 项目代码分析*
*随复现进度持续更新*
