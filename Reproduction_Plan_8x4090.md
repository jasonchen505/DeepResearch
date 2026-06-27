# DeepResearch 8×4090 全流程复现计划

> 硬件约束：8 × NVIDIA RTX 4090（24GB VRAM / 卡，共 192GB），PCIe 互联（无 NVLink）

---

## 一、资源约束与可行性分析

### 1.1 模型参数规格

| 模型 | 总参数 | 激活参数 | 原始精度权重大小 | 4-bit 量化后 |
|------|--------|---------|----------------|-------------|
| Tongyi-DeepResearch-30B-A3B | 30.5B | 3.3B/token | ~61 GB | ~16 GB |
| WebSailor-7B | 7B | 7B | ~14 GB | ~4 GB |
| WebSailor-3B | 3B | 3B | ~6 GB | ~2 GB |
| Qwen2.5-7B-Instruct (摘要模型) | 7B | 7B | ~14 GB | ~4 GB |

### 1.2 各阶段显存需求估算

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段            │ 方案              │ 单卡需求   │ 所需卡数  │ 可行性 │
├─────────────────────────────────────────────────────────────────────┤
│ 推理(30B BF16)  │ TP=4 分片         │ ~18GB     │ 4        │ ✓ 紧张 │
│ 推理(30B 4bit)  │ 单卡/TP=2         │ ~20GB     │ 1-2      │ ✓ 舒适 │
│ 推理(7B BF16)   │ 单卡              │ ~16GB     │ 1        │ ✓ 舒适 │
│ 推理(3B BF16)   │ 单卡              │ ~8GB      │ 1        │ ✓ 舒适 │
├─────────────────────────────────────────────────────────────────────┤
│ SFT(30B 全参)   │ BF16+Adam         │ ~80GB     │ 8+       │ ✗ 不行 │
│ SFT(30B QLoRA)  │ 4bit+LoRA r=64   │ ~20GB     │ 2-4      │ ✓ 可行 │
│ SFT(7B QLoRA)   │ 4bit+LoRA r=64   │ ~8GB      │ 1-2      │ ✓ 舒适 │
│ SFT(3B QLoRA)   │ 4bit+LoRA r=64   │ ~5GB      │ 1        │ ✓ 舒适 │
├─────────────────────────────────────────────────────────────────────┤
│ RL(30B GRPO)    │ 策略+参考+rollout │ ~50GB+    │ 8+       │ △ 极紧 │
│ RL(7B GRPO)     │ 4bit+LoRA        │ ~15GB     │ 2-4      │ ✓ 可行 │
│ RL(3B GRPO)     │ 4bit+LoRA        │ ~8GB      │ 1-2      │ ✓ 舒适 │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 核心结论

| 复现路径 | 可行性 | 说明 |
|---------|--------|------|
| **路径A：用官方30B模型做推理评估** | ✅ 可行 | 4-bit 量化后单卡可跑，4 卡并行可加速 |
| **路径B：基于小模型(3B/7B)全流程训练** | ✅ 可行 | SFT + RL 全流程可在 8 卡内完成 |
| **路径C：30B 模型 QLoRA SFT** | ✅ 可行 | 需要 4-8 卡，不能做全参 |
| **路径D：30B 模型 RL 全流程** | △ 困难 | 需要极端优化（量化+offload+gradient checkpoint） |
| **路径E：30B 全参 CPT** | ✗ 不可行 | 需要数百 GB 显存，超出 4090 能力 |

### 1.4 推荐路径

**采用「小模型全流程 + 大模型评估验证」的双轨策略：**

```
Phase 0: 环境搭建与数据准备
Phase 1: 30B 模型推理评估（验证系统可用性）
Phase 2: 3B/7B 模型 SFT 训练（学习训练流程）
Phase 3: 3B/7B 模型 RL 训练（学习 GRPO/DUPO）
Phase 4: 30B 模型 QLoRA 微调（进阶挑战）
Phase 5: 评估对比与迭代优化
```

---

## 二、Phase 0：环境搭建与数据准备（预计 1-2 天）

### 0.1 基础环境

```bash
# 创建 conda 环境
conda create -n deepresearch python=3.10.0
conda activate deepresearch

# 安装 CUDA 相关（4090 需要 CUDA 12.x）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
cd /data/home/yizhou/DeepResearch
pip install -r requirements.txt

# 额外训练依赖
pip install peft==0.15.0 bitsandbytes==0.45.0 deepspeed==0.16.0
pip install llama-factory  # SFT 训练框架
pip install verl           # RL 训练框架（WebDancer/WebSailor 使用）
```

### 0.2 模型下载

```bash
# 方案1: 使用 HuggingFace
huggingface-cli download Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --local-dir ./models/DeepResearch-30B
huggingface-cli download Alibaba-NLP/WebSailor-3B --local-dir ./models/WebSailor-3B
huggingface-cli download Alibaba-NLP/WebSailor-7B --local-dir ./models/WebSailor-7B

# 方案2: 使用 ModelScope（国内更快）
modelscope download --model iic/Tongyi-DeepResearch-30B-A3B --local_dir ./models/DeepResearch-30B
```

**磁盘空间需求**：
- DeepResearch-30B (BF16): ~61GB
- WebSailor-7B: ~14GB
- WebSailor-3B: ~6GB
- 总计: ~81GB（建议预留 150GB）

### 0.3 API Key 申请

| 服务 | 用途 | 申请地址 | 费用 |
|------|------|---------|------|
| Serper | Google 搜索 | serper.dev | 免费额度 2500 次 |
| Jina | 网页内容读取 | jina.ai | 免费额度 1M tokens |
| DashScope | 文件解析（可选） | dashscope.aliyun.com | 免费额度 |
| OpenAI API | 摘要模型（可选） | platform.openai.com | 按量付费 |

### 0.4 数据准备

```bash
# 项目自带的示例数据
ls WebAgent/WebDancer/datasets/
# sample_qa.jsonl    (200条 QA)
# sample_traj.jsonl  (200条 轨迹)

ls WebAgent/WebShaper/data/
# webshaper.500.jsonl (500条 形式化QA)

ls WebAgent/WebSailor/dataset/
# sailorfog-QA.jsonl

# 评估数据准备
mkdir -p eval_data
# 下载 GAIA benchmark（或其他 benchmark）
# 格式: {"question": "...", "answer": "..."}
```

### 0.5 环境配置文件

```bash
cp .env.example .env
# 编辑 .env 文件填入实际值
```

```bash
# .env 关键配置（4090 适配版）
MODEL_PATH=./models/WebSailor-7B   # 先用小模型验证
DATASET=eval_data/gaia.jsonl
OUTPUT_PATH=./outputs
ROLLOUT_COUNT=3
TEMPERATURE=0.6
PRESENCE_PENALTY=1.1
MAX_WORKERS=8                      # 4090 适当降低并发
SERPER_KEY_ID=your_key
JINA_API_KEYS=your_key
```

---

## 三、Phase 1：30B 模型推理评估（预计 2-3 天）

### 1.1 方案选择：4-bit 量化推理

**为什么选 4-bit 量化而非 BF16 Tensor Parallel？**

| 方案 | GPU数 | 吞吐量 | 配置复杂度 | 4090兼容性 |
|------|-------|--------|-----------|-----------|
| BF16 TP=4 | 4卡 | 中 | 中 | 紧张（需 NVLink 效果更好） |
| **GPTQ/AWQ 4-bit** | **1-2卡** | **中** | **低** | **✓ 最佳** |
| BF16 TP=8 | 8卡 | 高 | 高 | 4090 无 NVLink，TP 效率低 |

4090 没有 NVLink，Tensor Parallel 走 PCIe 效率很低。4-bit 量化后单卡 24GB 可以放下整个模型，是最佳方案。

### 1.2 使用 vLLM 进行 4-bit 量化推理

```bash
# 下载 GPTQ 量化版本（如果官方提供）
# 或者自行量化（需要额外 1 卡用于量化过程）

# 方案A: 使用 vLLM 加载 BF16 模型 + 自动量化
CUDA_VISIBLE_DEVICES=0 vllm serve ./models/DeepResearch-30B \
    --host 0.0.0.0 --port 6001 \
    --quantization awq \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90

# 方案B: 多卡 TP（如果单卡放不下）
CUDA_VISIBLE_DEVICES=0,1 vllm serve ./models/DeepResearch-30B \
    --host 0.0.0.0 --port 6001 \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 32768
```

### 1.3 适配 4090 的推理脚本

需要修改 `inference/run_react_infer.sh`：

```bash
#!/bin/bash
source ../.env

# 4090 适配：减少并发实例数
# 原版用 8 卡各跑 1 个实例，4090 改为 4 个实例（2卡/实例用 TP）
echo "Starting VLLM servers (4090 optimized)..."

# 实例1: GPU 0,1 (TP=2)
CUDA_VISIBLE_DEVICES=0,1 vllm serve $MODEL_PATH \
    --host 0.0.0.0 --port 6001 \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 32768 \
    --disable-log-requests &

# 实例2: GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 vllm serve $MODEL_PATH \
    --host 0.0.0.0 --port 6002 \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 32768 \
    --disable-log-requests &

# 实例3: GPU 4,5
CUDA_VISIBLE_DEVICES=4,5 vllm serve $MODEL_PATH \
    --host 0.0.0.0 --port 6003 \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 32768 \
    --disable-log-requests &

# 实例4: GPU 6,7
CUDA_VISIBLE_DEVICES=6,7 vllm serve $MODEL_PATH \
    --host 0.0.0.0 --port 6004 \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 32768 \
    --disable-log-requests &

# 等待服务就绪...
# 修改 planning_ports 为 [6001, 6002, 6003, 6004]

# 开始推理
python -u run_multi_react.py \
    --dataset "$DATASET" \
    --output "$OUTPUT_PATH" \
    --max_workers 8 \          # 4090 降低并发
    --model $MODEL_PATH \
    --temperature 0.6 \
    --presence_penalty 1.1 \
    --roll_out_count 3
```

### 1.4 快速验证：先用 WebSailor-3B 跑通

```bash
# 最小成本验证系统可用性
CUDA_VISIBLE_DEVICES=0 vllm serve ./models/WebSailor-3B \
    --host 0.0.0.0 --port 6001 \
    --max-model-len 16384

# 用少量数据测试
python run_multi_react.py \
    --dataset eval_data/test_10.jsonl \
    --output ./outputs/test \
    --max_workers 2 \
    --model ./models/WebSailor-3B \
    --roll_out_count 1
```

### 1.5 评估脚本

```bash
# 运行评估
cd evaluation
python evaluate_deepsearch_official.py \
    --input_fp ../outputs/WebSailor-3B/gaia \
    --dataset gaia
```

---

## 四、Phase 2：小模型 SFT 训练（预计 3-5 天）

### 4.1 为什么从 SFT 开始

- SFT 是 RL 的前置条件（冷启动）
- SFT 的技术门槛相对低，适合先熟悉训练流程
- 项目提供了现成的 SFT 数据（`sample_traj.jsonl`）

### 4.2 训练数据格式转换

项目提供的 `sample_traj.jsonl` 是多轮对话格式，需要转换为 SFT 训练格式：

```python
# convert_traj_to_sft.py
import json

def convert_trajectory_to_sft(input_file, output_file):
    """将 Agent 轨迹转换为 SFT 训练格式"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            messages = data['messages']
            
            # 过滤掉失败的轨迹（没有 <answer> 的）
            has_answer = any('<answer>' in m.get('content', '') 
                          for m in messages if m['role'] == 'assistant')
            if not has_answer:
                continue
            
            # 写入训练格式
            sft_sample = {"messages": messages}
            f_out.write(json.dumps(sft_sample, ensure_ascii=False) + '\n')

convert_trajectory_to_sft(
    'WebAgent/WebDancer/datasets/sample_traj.jsonl',
    'train_data/sft_train.jsonl'
)
```

### 4.3 方案A：使用 LLaMA-Factory 做 QLoRA SFT

WebDancer 论文明确提到使用 LLaMA-Factory 做 SFT。

```yaml
# llama_factory_config.yaml (4090 适配)
### model
model_name_or_path: ./models/WebSailor-7B
quantization_bit: 4           # 4-bit QLoRA
quantization_method: bitsandbytes

### method
stage: sft
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all              # 对所有 linear 层做 LoRA
lora_dropout: 0.05

### dataset
dataset: deepresearch_traj    # 注册自定义数据集
template: qwen2.5             # Qwen 系列模板
cutoff_len: 32768             # Agent 轨迹很长，需要大 cutoff

### output
output_dir: ./output_sft_webagent_7b
logging_steps: 10
save_steps: 500

### train
per_device_train_batch_size: 1       # 4090 显存有限
gradient_accumulation_steps: 16      # 等效 batch_size=16
num_train_epochs: 3
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true                           # 4090 支持 BF16

### 4090 显存优化
gradient_checkpointing: true         # 节省显存
flash_attn: true                     # 使用 Flash Attention
```

```bash
# 启动 SFT 训练
llamafactory-cli train llama_factory_config.yaml
```

### 4.4 方案B：使用 DeepSpeed ZeRO-3 + QLoRA 多卡训练

```bash
# deepspeed_config_sft.json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e6
    },
    "gradient_accumulation_steps": 16,
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_checkpointing": {"enabled": true}
}
```

```bash
# 多卡 SFT 训练
deepspeed --num_gpus 8 train_sft.py \
    --model_name ./models/WebSailor-7B \
    --data_path train_data/sft_train.jsonl \
    --output_dir ./output_sft \
    --lora_rank 64 \
    --deepspeed deepspeed_config_sft.json
```

### 4.5 SFT 训练监控要点

```python
# 关键监控指标
metrics_to_watch = {
    "train_loss": "应稳步下降，不应震荡",
    "learning_rate": "cosine schedule，先warmup后decay",
    "gpu_memory_usage": "不应超过 22GB（4090 安全线）",
    "grad_norm": "梯度范数，不应爆炸（>10 要注意）",
}

# 显存监控
import torch
print(f"GPU 0 allocated: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
print(f"GPU 0 reserved: {torch.cuda.memory_reserved(0)/1e9:.1f}GB")
```

---

## 五、Phase 3：小模型 RL 训练（预计 3-5 天）

### 5.1 RL 训练资源需求分析

GRPO 训练需要同时维护：
1. **Policy Model**（可训练参数）：需要梯度
2. **Reference Model**（冻结参数）：不需要梯度，但需要在内存中
3. **Rollout Buffer**：存储采样的轨迹

**WebSailor-7B 的显存估算（QLoRA + GRPO）**：
- 模型权重(4bit): ~4GB
- LoRA 参数(BF16): ~0.2GB
- 优化器状态: ~0.8GB
- 梯度: ~0.4GB
- Reference Model(4bit): ~4GB
- KV Cache + Activations: ~4-8GB
- **总计: ~14-18GB / 卡**（可接受）

### 5.2 GRPO 训练配置

```python
# grpo_config.py
import verl

config = {
    "model": {
        "path": "./output_sft_webagent_7b/merged",  # SFT 后的模型
        "ref_model_path": "./output_sft_webagent_7b/merged",
        "enable_lora": True,
        "lora_rank": 64,
        "quantization": "4bit",
    },
    "training": {
        "algorithm": "grpo",
        "rollout_count": 3,           # 每个 prompt 采样 3 个 response
        "max_response_len": 32768,    # Agent 轨迹最大长度
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 1.1,
        "learning_rate": 1e-6,        # RL 学习率通常很小
        "kl_penalty_coef": 0.01,      # KL 散度惩罚系数
        "clip_range": 0.2,            # PPO clip range
        "batch_size": 8,              # 每批 8 个 prompt
        "mini_batch_size": 4,
    },
    "env": {
        "tool_server": "http://localhost:8080",  # 工具服务地址
        "max_tool_calls": 100,
        "timeout_per_call": 200,
    },
    "hardware": {
        "num_gpus": 8,
        "offload_ref_model": True,     # 将 ref model offload 到 CPU
        "offload_optimizer": True,     # 将优化器 offload 到 CPU
    },
}
```

### 5.3 RL 训练的工具服务

RL 训练中，Agent 需要与环境交互（调用搜索、访问网页）。需要启动一个工具服务：

```bash
# 启动工具服务（独立进程）
python -m tool_server \
    --port 8080 \
    --serper_key $SERPER_KEY_ID \
    --jina_key $JINA_API_KEYS
```

### 5.4 DUPO 训练（WebSailor 的 RL 算法）

WebSailor 提出 DUPO (Duplicating Sampling Policy Optimization)，相比标准 GRPO：
- 在 RFT 冷启动后使用
- 对同一 prompt 进行重复采样以提高效率
- 结合 selective filtering 稳定训练

```bash
# DUPO 训练脚本
python train_dupo.py \
    --model_path ./output_sft_webagent_7b/merged \
    --data_path train_data/rl_prompts.jsonl \
    --output_dir ./output_rl_webagent_7b \
    --num_gpus 8 \
    --rollout_count 3 \
    --max_steps 1000 \
    --save_steps 200
```

### 5.5 RL 训练监控

```python
# 关键监控
rl_metrics = {
    "reward_mean": "平均 reward，应逐步上升",
    "reward_std": "reward 方差，反映探索程度",
    "kl_divergence": "策略偏离程度，不应过大（<0.1）",
    "policy_loss": "策略损失",
    "entropy": "策略熵，过低说明探索不足",
    "pass_at_1": "在验证集上的 Pass@1",
    "avg_trajectory_length": "平均轨迹长度，过长说明效率低",
}
```

---

## 六、Phase 4：30B 模型 QLoRA 微调（进阶，预计 3-5 天）

### 6.1 30B QLoRA 显存分析

```
模型权重(4bit):    ~16 GB
LoRA 参数(r=64):   ~0.5 GB (BF16)
优化器状态:        ~2 GB (Adam, offload to CPU)
梯度:              ~1 GB (offload to CPU)
激活值:            ~4-8 GB (gradient checkpointing)
─────────────────────────
总计:              ~24 GB / 卡 (刚好适配 4090)
```

### 6.2 DeepSpeed ZeRO-3 + QLoRA 配置

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "gradient_accumulation_steps": 32,
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_checkpointing": {"enabled": true},
    "wall_clock_breakdown": true
}
```

### 6.3 30B SFT 训练命令

```bash
deepspeed --num_gpus 8 train_sft.py \
    --model_name ./models/DeepResearch-30B \
    --data_path train_data/sft_train.jsonl \
    --output_dir ./output_sft_30b \
    --lora_rank 64 \
    --lora_alpha 128 \
    --quantization 4bit \
    --cutoff_len 32768 \
    --deepspeed deepspeed_config_30b.json \
    --gradient_checkpointing \
    --per_device_batch_size 1 \
    --num_epochs 2 \
    --lr 1e-4
```

### 6.4 30B 模型合并与导出

```python
# merge_lora.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "./models/DeepResearch-30B",
    torch_dtype="auto",
    device_map="auto"
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "./output_sft_30b/checkpoint-best")

# 合并
model = model.merge_and_unload()

# 保存
model.save_pretrained("./models/DeepResearch-30B-SFT")
tokenizer = AutoTokenizer.from_pretrained("./models/DeepResearch-30B")
tokenizer.save_pretrained("./models/DeepResearch-30B-SFT")
```

---

## 七、Phase 5：评估对比与迭代（预计 2-3 天）

### 7.1 评估矩阵

| 模型 | GAIA | WebWalkerQA | BrowseComp | 平均延迟 |
|------|------|-------------|------------|---------|
| DeepResearch-30B (官方) | baseline | baseline | baseline | - |
| WebSailor-7B (SFT后) | 待测 | 待测 | 待测 | - |
| WebSailor-7B (RL后) | 待测 | 待测 | 待测 | - |
| WebSailor-3B (SFT后) | 待测 | 待测 | 待测 | - |

### 7.2 评估脚本

```bash
# 评估所有模型
for model in WebSailor-3B WebSailor-7B DeepResearch-30B-SFT; do
    echo "Evaluating $model..."
    
    # 启动推理服务
    CUDA_VISIBLE_DEVICES=0 vllm serve ./models/$model \
        --host 0.0.0.0 --port 6001 &
    sleep 60  # 等待加载
    
    # 运行推理
    python inference/run_multi_react.py \
        --dataset eval_data/gaia.jsonl \
        --output ./outputs/$model \
        --model ./models/$model \
        --max_workers 4 \
        --roll_out_count 3
    
    # 运行评估
    python evaluation/evaluate_deepsearch_official.py \
        --input_fp ./outputs/$model \
        --dataset gaia
    
    # 关闭服务
    pkill -f "vllm serve"
    sleep 10
done
```

### 7.3 迭代优化方向

根据评估结果，针对性优化：

| 问题 | 优化方向 |
|------|---------|
| Pass@1 低 | 增加 SFT 数据量/质量、改进 system prompt |
| 工具调用失败率高 | 检查 API 配额、增加重试逻辑 |
| 超时率高 | 优化 max_workers、增加超时时间 |
| 格式错误多 | 在 SFT 数据中增加格式正确的样本 |
| RL 不收敛 | 调整学习率、KL 系数、rollout 数量 |

---

## 八、时间线总结

```
Week 1: Phase 0 (环境搭建) + Phase 1 (30B 推理评估)
  ├── Day 1-2: 环境搭建、模型下载、API 申请
  ├── Day 3-4: WebSailor-3B 快速验证跑通全流程
  └── Day 5-7: 30B 量化推理、评估 benchmark

Week 2: Phase 2 (SFT 训练)
  ├── Day 8-9: 数据准备、格式转换
  ├── Day 10-11: WebSailor-7B QLoRA SFT
  └── Day 12-14: SFT 评估、数据清洗迭代

Week 3: Phase 3 (RL 训练)
  ├── Day 15-16: GRPO/DUPO 环境配置
  ├── Day 17-19: WebSailor-7B RL 训练
  └── Day 20-21: RL 评估、调参

Week 4: Phase 4 + Phase 5 (进阶 + 评估)
  ├── Day 22-24: 30B QLoRA 微调
  ├── Day 25-26: 全模型评估对比
  └── Day 27-28: 总结报告、代码整理
```

---

## 九、风险与应对

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| 30B 量化后精度损失大 | 中 | 高 | 改用 BF16 TP=4，或降级到 7B 模型 |
| 4090 OOM | 中 | 中 | 降低 batch_size、增加 gradient accumulation |
| API 额度耗尽 | 高 | 中 | 使用缓存、降低 rollout 数量 |
| RL 训练不收敛 | 中 | 高 | 先用小数据集验证、调整超参 |
| 评估数据不可用 | 低 | 高 | 使用项目自带的 sample 数据 |

---

## 十、关键文件路径速查

| 用途 | 文件 |
|------|------|
| 推理主循环 | `inference/react_agent.py` |
| 多 Rollout 调度 | `inference/run_multi_react.py` |
| 推理启动脚本 | `inference/run_react_infer.sh` |
| System Prompt | `inference/prompt.py` |
| SFT 训练数据 | `WebAgent/WebDancer/datasets/sample_traj.jsonl` |
| QA 数据 | `WebAgent/WebDancer/datasets/sample_qa.jsonl` |
| WebShaper 数据 | `WebAgent/WebShaper/data/webshaper.500.jsonl` |
| SailorFog 数据 | `WebAgent/WebSailor/dataset/sailorfog-QA.jsonl` |
| 评估脚本 | `evaluation/evaluate_deepsearch_official.py` |
| 环境配置 | `.env.example` |

---

*创建日期: 2026-06-27*
*硬件环境: 8 × NVIDIA RTX 4090 (24GB)*
