# 2026年主流大模型横向对比报告

> 整理时间：2026年5月  
> 覆盖模型：Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.6-Plus、DeepSeek V4-Pro、DeepSeek V4-Flash、Claude Opus 4.7、GPT-5.5

---

## 目录

1. [模型概览](#1-模型概览)
2. [架构与参数详解](#2-架构与参数详解)
3. [训练流程](#3-训练流程)
4. [Benchmark 完整对比](#4-benchmark-完整对比)
5. [部署参数与显存需求](#5-部署参数与显存需求)
6. [定价对比](#6-定价对比)
7. [各模型优缺点与定位](#7-各模型优缺点与定位)
8. [选型建议](#8-选型建议)
9. [关键概念说明](#9-关键概念说明)

---

## 1. 模型概览

| 模型 | 发布方 | 发布时间 | 架构 | 总参数 | 激活参数 | 上下文 | 开源 | 许可证 |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-27B** | Alibaba | 2026.04.22 | Dense | 27B | 27B（全） | 1M | ✅ | Apache 2.0 |
| **Qwen3.6-35B-A3B** | Alibaba | 2026.04.14 | MoE | 35B | 3B | 1M | ✅ | Apache 2.0 |
| **Qwen3.6-Plus** | Alibaba | 2026.04（Preview）| 混合（未公开）| 未公开 | 未公开 | 1M | ❌ | 商用 API |
| **DeepSeek V4-Pro** | DeepSeek | 2026.04.24 | MoE | 1.6T | 49B | 1M | ✅ | MIT |
| **DeepSeek V4-Flash** | DeepSeek | 2026.04.24 | MoE | 284B | 13B | 1M | ✅ | MIT |
| **Claude Opus 4.7** | Anthropic | 2026初 | 闭源 | 未公开 | — | 1M | ❌ | 商用 API |
| **GPT-5.5** | OpenAI | 2026.03 | 闭源 | 未公开 | — | 1.05M | ❌ | 商用 API |

---

## 2. 架构与参数详解

### 2.1 Qwen3.6-27B（全密集旗舰）

**核心定位**：用 27B 密集参数超越上代 397B MoE 旗舰（Qwen3.5-397B-A17B）。

| 参数 | 值 |
|---|---|
| 层数 | 64 |
| 隐藏维度 | 5120 |
| FFN 中间维度 | 17,408 |
| 词汇表大小 | 248,320 |
| DeltaNet 头（V/QK） | 48 / 16（头维度 128）|
| Gated Attention 头（Q/KV） | 24 / 4（头维度 256）|
| RoPE 旋转维度 | 64 |
| 原生上下文 | 262,144 tokens |
| 最大上下文（YaRN） | 1,010,000 tokens |
| 多模态 | ✅ 文本 + 图像 + 视频 |
| MTP 投机解码 | ✅ |
| 思维保留 | ✅ |

**层结构**（循环 16 次）：
```
3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN)
```

### 2.2 Qwen3.6-35B-A3B（高速 MoE）

**核心定位**：3B 激活算力实现 35B 知识量，速度比 27B 快 3-4 倍。

| 参数 | 值 |
|---|---|
| 层数 | 40 |
| 隐藏维度 | 2048 |
| 词汇表大小 | 248,320 |
| 专家总数 | 256 |
| 路由专家数 | 8（+ 1 共享）= 9 激活 |
| 专家中间维度 | 512 |
| DeltaNet 头（V/QK） | 32 / 16（头维度 128）|
| Gated Attention 头（Q/KV） | 16 / 2（头维度 256）|
| 原生上下文 | 262,144 tokens |
| 最大上下文（YaRN） | 1,010,000 tokens |
| 多模态 | ✅ 文本 + 图像 + 视频 |
| MTP 投机解码 | ✅ |

**层结构**（循环 10 次，每层接 MoE FFN）：
```
3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)
```

### 2.3 Qwen3.6 共用架构创新：Gated DeltaNet

两款 Qwen3.6 模型均采用 **Gated DeltaNet 线性注意力**作为主干（占 3/4 层）：

```
传统 Self-Attention：O(n²) 复杂度，长序列极慢
DeltaNet 线性注意力：O(n) 复杂度，长序列高效
Gated DeltaNet = DeltaNet + LSTM 式门控机制
  → 学习"何时更新、何时保留"信息
  → 3:1 混合比例：3 层 DeltaNet + 1 层标准 GatedAttention
  → DeltaNet 负责长序列效率，标准注意力补足精确信息检索
```

### 2.4 Qwen3.6-Plus（闭源 API 旗舰）

| 参数 | 值 |
|---|---|
| 参数量 | 未公开 |
| 上下文 | 1M tokens |
| 最大输出 | 65,536 tokens |
| 思维链 | **强制开启**（不可关闭，与其他模型不同）|
| 多模态 | ❌ 纯文本 |
| 状态 | Preview（免费）|

### 2.5 DeepSeek V4-Pro / V4-Flash

**核心定位**：面向 Agent 场景的百万上下文 MoE，以极低算力需求运行超长上下文。

| 参数 | V4-Pro | V4-Flash |
|---|---|---|
| 总参数 | **1.6T** | 284B |
| 激活参数 | 49B | 13B |
| 层数 | 61 | 未公开 |
| 预训练数据 | 33T tokens | 32T tokens |
| 上下文 | 1M | 1M |
| 推理模式 | Non-Think / Think High / **Think Max** | 同左 |
| 精度 | FP4（专家）+ FP8（其余）| 同左 |

**DeepSeek V4 架构创新**（相比 Qwen3.6 完全不同的技术路线）：

```
核心问题：1M token 下 KV Cache 显存爆炸

解法：三层压缩注意力架构（SALS）

HCA（Heavily Compressed Attention）
  → KV 压缩 128x，对所有压缩块做全量注意力

CSA（Compressed Sparse Attention）
  → KV 压缩 4x（softmax 门控池化）
  → FP4 闪电索引器选 top-k 相关块
  → 稀疏 × 压缩双重节省

层排布（V4-Pro 61 层）：
  层 0-1: HCA
  层 2-60: CSA 与 HCA 交替
  末尾 MTP 块: 滑动窗口注意力
```

**1M token 下的效率（对比 V3.2）**：
- 推理 FLOPs：仅需 **27%**
- KV Cache：仅需 **10%**（对比标准 GQA 仅需 **2%**）

**其他训练创新**：

| 创新 | 作用 |
|---|---|
| mHC（流形约束超连接） | 1.6T 参数下稳定梯度传播 |
| Muon 优化器 | 二阶优化，比 AdamW 收敛更快更稳 |
| MTP 4 步预测 | 训练时多步预测，推理时投机解码加速 |

---

## 3. 训练流程

Qwen3.6 与 DeepSeek V4 均采用多阶段后训练，核心流程如下：

### Qwen 四阶段训练管线

```
阶段 1：长 CoT 冷启动
  → 大规模长链式思维数据 SFT
  → 覆盖数学、代码、逻辑、STEM
  → 建立基础推理能力

阶段 2：推理强化学习（RL）
  → 规则化奖励（rule-based rewards）
  → 大规模算力扩展
  → 增强探索与利用能力

阶段 3：思维模式融合
  → 将"非思考"能力注入思考模型
  → 混合长 CoT + 常规指令数据
  → 实现思考/非思考模式统一切换

阶段 4：通用 RL 对齐
  → 覆盖 20+ 通用任务领域
  → 修正指令遵循、格式、Agent 行为
```

### DeepSeek V4 训练特点

```
预训练：32-33T tokens（超大规模语料）
后训练（两阶段）：
  阶段 1：独立培养领域专家（SFT + GRPO RL）
  阶段 2：统一模型整合（在线蒸馏，融合各领域能力）
优化器：Muon（二阶方法）替代 AdamW
稳定性：mHC 超连接防止 1.6T 规模的梯度消失
```

---

## 4. Benchmark 完整对比

> 说明：DeepSeek 数字均为 **Max 模式**（最高推理预算），Qwen3.6 为思考模式，Claude/GPT 为最高可用模式。

### 4.1 编程 Agent

| Benchmark | Qwen3.6-27B | Qwen3.6-35B-A3B | DS V4-Flash | DS V4-Pro | Claude Opus 4.6 | GPT-5.4 |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 77.2 | 73.4 | 79.0 | 80.6 | **80.8** | — |
| SWE-bench Pro | 53.5 | 49.5 | 52.6 | 55.4 | 57.3 | **57.7** |
| SWE-bench Multilingual | 71.3 | 67.2 | 73.3 | 76.2 | **77.5** | — |
| Terminal-Bench 2.0 | 59.3 | 51.5 | 56.9 | 67.9 | 65.4 | **75.1** |
| LiveCodeBench v6 | 83.9 | 80.4 | 91.6 | **93.5** | 88.8 | — |
| Codeforces Rating | — | — | 3052 | **3206** | — | 3168 |
| MCPAtlas Public | — | 62.8 | 69.0 | 73.6 | **73.8** | 67.2 |
| Toolathlon | — | 26.9 | 47.8 | 51.8 | 47.2 | **54.6** |
| BrowseComp | — | — | 73.2 | 83.4 | **83.7** | 82.7 |
| SkillsBench Avg5 | **48.2** | 28.7 | — | — | 45.3 | — |

> **注**：Qwen3.6-27B 的 SkillsBench **48.2** 超越 Claude 4.5 Opus（45.3）；Terminal-Bench 2.0 中 27B 追平 Claude（均 59.3）。

### 4.2 推理与知识

| Benchmark | Qwen3.6-27B | Qwen3.6-35B-A3B | DS V4-Flash | DS V4-Pro | Claude Opus 4.6 | GPT-5.4 |
|---|---|---|---|---|---|---|
| GPQA Diamond | 87.8 | 86.0 | 88.1 | 90.1 | 91.3 | **93.0** |
| HLE（无工具） | 24.0 | 21.4 | 34.8 | 37.7 | 40.0 | 39.8 |
| HLE（带工具） | — | — | 45.1 | 48.2 | **53.1** | 52.0 |
| MMLU-Pro | 86.2 | 85.2 | 86.2 | 87.5 | **89.1** | 87.5 |
| HMMT Feb 26 | 84.3 | 83.6 | 94.8 | 95.2 | 96.2 | **97.7** |
| IMOAnswerBench | 80.8 | 78.9 | 88.4 | 89.8 | 75.3 | **91.4** |
| SimpleQA-Verified | — | — | 34.1 | 57.9 | 46.2 | 45.3 |
| AIME 2026 | 94.1 | 92.7 | — | ~96.4 | 93.3 | ~96.7 |

### 4.3 长上下文

| Benchmark | Qwen3.6-27B | DS V4-Flash | DS V4-Pro | Claude Opus 4.6 | Gemini 3.1 Pro |
|---|---|---|---|---|---|
| MRCR 1M | — | 78.7 | 83.5 | **92.9** | 76.3 |
| CorpusQA 1M | — | 60.5 | 62.0 | **71.7** | 53.8 |
| 1M Context 可靠性 | — | — | **97.0%** | — | 94.0% |

### 4.4 DS V4 各推理模式分层数据

| Benchmark | Flash Non | Flash High | Flash Max | Pro Non | Pro High | Pro Max |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 73.7 | 78.6 | **79.0** | 73.6 | 79.4 | **80.6** |
| Terminal-Bench 2.0 | 49.1 | 56.6 | **56.9** | 59.1 | 63.3 | **67.9** |
| GPQA Diamond | 71.2 | 87.4 | **88.1** | 72.9 | 89.1 | **90.1** |
| LiveCodeBench | 55.2 | 88.4 | **91.6** | 56.8 | 89.8 | **93.5** |
| Codeforces Rating | — | 2816 | **3052** | — | 2919 | **3206** |
| HMMT Feb 26 | 40.8 | 91.9 | **94.8** | 31.7 | 94.0 | **95.2** |
| HLE | 8.1 | 29.4 | **34.8** | 7.7 | 34.5 | **37.7** |
| MRCR 1M | 37.5 | 76.9 | **78.7** | 44.7 | 83.3 | **83.5** |
| MCPAtlas | 64.0 | 67.4 | **69.0** | 69.4 | 74.2 | **73.6** |

**Flash Max vs Pro Max 实际差距**：

```
SWE-bench Verified:  79.0 vs 80.6  → 差 1.6%，日常编程基本持平
LiveCodeBench:       91.6 vs 93.5  → 差 1.9%
GPQA Diamond:        88.1 vs 90.1  → 差 2.0%
Terminal-Bench 2.0:  56.9 vs 67.9  → 差 11%，多步 Agent 有明显差距
SimpleQA-Verified:   34.1 vs 57.9  → 差 23.8%，事实知识召回明显弱
```

---

## 5. 部署参数与显存需求

### 5.1 激活参数 ≠ 显存需求

这是 MoE 模型最常见的误解：

- **激活参数**：决定每次推理的**算力（FLOPs）**
- **显存需求**：由**全部参数**决定——所有专家权重必须预先加载，路由器才能随时选用任意专家

### 5.2 显存估算

| 模型 | 总参数 | 精度 | **实际显存需求** | 最低配置 |
|---|---|---|---|---|
| Qwen3.6-27B | 27B | BF16 | ~54 GB | 2× RTX 4090 |
| Qwen3.6-27B | 27B | FP8 | ~27 GB | 1× A100 80GB |
| Qwen3.6-35B-A3B | 35B | BF16 | ~70 GB | 2× A100 |
| Qwen3.6-35B-A3B | 35B | Q4 量化 | **~21 GB** | 1× RTX 3090 ✅ |
| DS V4-Flash | 284B | FP4+FP8 | **~170 GB+** | 4-6× H100 |
| DS V4-Pro | 1.6T | FP4+FP8 | **~900 GB+** | 10-12× H100 |

### 5.3 DS V4-Flash 的 CPU 卸载方案

KTransformers 等框架支持将非激活专家卸载到内存（RAM）：

```
极限省显存配置：
  GPU VRAM：~24 GB（仅放激活参数 + 注意力层）
  系统内存：~200 GB+
  代价：受 PCIe 带宽限制，推理速度极慢
  适合：本地研究，不适合生产
```

### 5.4 各模型推荐部署栈

| 模型 | 推荐框架 | 关键参数 |
|---|---|---|
| Qwen3.6-27B | vLLM ≥ 0.19.0 / SGLang ≥ 0.5.10 | `--kv-cache-dtype fp8 --enable-prefix-caching --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| Qwen3.6-35B-A3B | vLLM / KTransformers | `--tensor-parallel-size 2 --max-model-len 262144` |
| DS V4-Flash/Pro | vLLM / SGLang（多节点） | `temperature=1.0, top_p=1.0`；Think Max 需 384K+ ctx |

### 5.5 Qwen3.6-27B 极致速度优化（视频实测 20→184 t/s）

叠加以下方法可将生成速度从 20 t/s 提升至 184 t/s：

| 优化手段 | 命令/方式 | 效果 |
|---|---|---|
| 张量并行 | `--tensor-parallel-size 2` | 基础多卡扩展 |
| FP8 KV 量化 | `--kv-cache-dtype fp8` | 内存减半，并发 4x→7x |
| MTP 投机解码 | `--speculative-config mtp, num=3` | 约 1.9x 加速 |
| NVFP4 量化 | Blackwell GPU 专用 | BF16 51GB→NVFP4 26GB |
| Prefix Caching | `--enable-prefix-caching` | 长对话 KV 复用 |
| Chunked Prefill | `--enable-chunked-prefill` | 降低首 token 延迟 |

---

## 6. 定价对比

| 模型 | 输入价格 | 输出价格 | 缓存命中 | 备注 |
|---|---|---|---|---|
| **Qwen3.6-Plus** | 免费 | 免费 | — | Preview 期，收集数据 |
| **DS V4-Flash** | $0.14/M | $0.28/M | — | 开源可自部署 |
| **DS V4-Pro** | $1.74/M | $3.48/M | $0.145/M | 开源可自部署 |
| **GPT-5.4** | ~$2.50/M | ~$15/M | — | 闭源 |
| **Claude Opus 4.7** | $5/M | $25/M | — | 闭源 |

> DS V4-Flash 在 API 成本上碾压闭源模型：Claude Opus 输出价格是它的 **89 倍**。

---

## 7. 各模型优缺点与定位

### Qwen3.6-27B

**优势**
- 开源最强密集模型，27B 参数超越上代 397B MoE
- 编程 Agent 追平 Claude Opus（部分超越）：SkillsBench 48.2 > Claude 45.3
- 18GB FP8 显存即可运行，单张 A100 可承载
- 原生多模态（文本+图像+视频）
- Apache 2.0，无商用限制

**劣势**
- 推理速度慢于 35B-A3B（全密集，每次激活 27B）
- 世界知识广度（MMLU、HLE）落后闭源模型约 5-10%
- 无法比拟 GPT-5.5/Claude 的 Agent 生态成熟度

---

### Qwen3.6-35B-A3B

**优势**
- 速度 3-4x 快于 27B（仅激活 3B 参数）
- Q4 量化仅需 21GB 显存，MacBook M5 可运行
- 编程能力与 Claude Sonnet 4.5 持平或超越
- MCP 工具调用（MCPMark 37.0，同级最强）

**劣势**
- 编程质量被 27B 全面超越（SkillsBench 28.7 vs 48.2）
- 质量上限低于 Dense 27B
- MoE 路由增加部署复杂度

---

### Qwen3.6-Plus

**优势**
- 1M 上下文，Preview 期免费
- 思维链强制开启，输出质量稳定
- 速度比 Claude Opus 快 2-3 倍（DeepSeek 官方集群）

**劣势**
- 参数量完全不透明
- 生产可靠性未经充分验证
- 纯文本，无多模态
- 思维链不可关闭，不适合低延迟场景

---

### DeepSeek V4-Pro

**优势**
- 开源最强综合模型，MIT 许可
- LiveCodeBench **93.5%** 超越所有对标模型
- Codeforces **3206** 超越 GPT-5.4（3168）
- Terminal-Bench **67.9%** 超越 Claude Opus 4.6（65.4%）
- 1M 上下文可靠性 **97%**，远超 GPT-5.5（82.5%）
- 比 Claude Opus 便宜约 **7 倍**

**劣势**
- 总参数 1.6T，本地部署需要 10+ 张 H100
- 知识精确度（SimpleQA 57.9 vs Gemini 75.6）有差距
- SWE-bench Pro（55.4）略落后于 Claude（57.3）和 K2.6（58.6）
- 无多模态

---

### DeepSeek V4-Flash

**优势**
- SWE-bench 79.0%，追平上代 Claude Opus（差 1.6%）
- 仅 **$0.14/M** 输入，Claude Opus 的 1/35
- 1M 上下文，MIT 开源
- 简单 Agent 任务与 Pro 几乎持平

**劣势**
- 总参数 284B，本地部署仍需 4-6 张 H100
- Terminal-Bench 56.9%，比 Pro 低 11%，多步 Agent 有明显差距
- 事实知识（SimpleQA 34.1 vs Pro 57.9）明显弱

---

### Claude Opus 4.7

**优势**
- Tau2-Bench Agent 能力 **91.6%**，仍是最强通用 Agent
- MRCR 1M 长文检索 **92.9%**，远超 DeepSeek V4-Pro（83.5%）
- HLE 带工具 **53.1%**，最强复杂推理+工具组合
- 工具链生态最成熟，企业级 SLA

**劣势**
- 最贵（$5/M 输入，$25/M 输出）
- 闭源，数据隐私需额外评估
- LiveCodeBench 被 DS V4-Pro 超越（88.8 vs 93.5）

---

### GPT-5.5

**优势**
- 综合智力指数第一（Intelligence Index 60）
- Terminal-Bench **82.7%**，领先所有模型
- 数学推理（HMMT 97.7%、IMOAnswerBench 91.4%）最强
- 工具生态最成熟

**劣势**
- 完全闭源，参数未公开
- 定价偏高
- 长上下文可靠性（82.5%）不及 DS V4-Pro（97%）

---

## 8. 选型建议

```
极致编程质量 + 开源自托管（有 A100）
  → Qwen3.6-27B
  → 理由：编程 Agent 接近顶级闭源，Apache 2.0，18GB FP8 可运行

速度优先 + 消费级 GPU 本地部署
  → Qwen3.6-35B-A3B
  → 理由：Q4 量化 21GB，速度 3-4x，MacBook M5 可跑

高性价比 API + 日常编程任务
  → DeepSeek V4-Flash
  → 理由：SWE-bench 79%，$0.14/M，Claude 的 1/35 价格

最强编程 API + 长上下文 + 开源
  → DeepSeek V4-Pro
  → 理由：LiveCodeBench 93.5% 第一，1M ctx 可靠性 97%，MIT 开源
  → 代价：需要 10+ 张 H100，不适合本地部署

最强通用 Agent + 企业级可靠
  → Claude Opus 4.7
  → 理由：Tau2-Bench 91.6% 第一，长文检索最强，生态最成熟

综合天花板 + 数学推理
  → GPT-5.5
  → 理由：Terminal-Bench 82.7% 第一，HMMT/IMO 数学最强

超长上下文文档处理（免费）
  → Qwen3.6-Plus（Preview 期）
  → 理由：1M ctx，免费，速度快；生产稳定性待验证
```

---

## 9. 关键概念说明

### Terminal-Bench vs 通用 Agent Benchmark

Terminal-Bench 2.0 和 Tau2/TAU3-Bench 测量的是不同维度的 Agent 能力：

| | **Terminal-Bench 2.0** | **TAU3-Bench / Tau2-Bench** |
|---|---|---|
| 场景 | Linux 终端 + bash + 文件编辑 | 客服、电商、订票、网页操作 |
| 工具 | shell、文件系统、代码执行 | 数据库、API、表单填写 |
| 超时 | 3 小时，32CPU/48GB RAM | 无极端资源限制 |
| 评分 | 二元（成功/失败）| 多维度 |

Qwen3.6 系列在 Terminal-Bench（编程专项）接近顶级闭源，但在 TAU3-Bench（通用业务 Agent）优势消失——Qwen3.6 是编程专才，Claude 是通用 Agent 全才。

### 激活参数 vs 总参数 vs 显存

```
激活参数 → 每次推理的算力消耗（FLOPs）
总参数   → 实际占用的显存（所有专家必须预加载）
显存需求 ≈ 总参数 × 每参数字节数

示例：
  DS V4-Flash（284B 总 / 13B 激活）
    显存 ≈ 284B × 0.6 bytes ≈ 170GB
    但算力 ≈ 仅 13B 参数的 Dense 模型
```

### MTP 投机解码

Multi-Token Prediction：模型训练时同时预测多个未来 token，推理时用多步预测头生成草稿，主模型批量验证，提升吞吐而不损失质量。Qwen3.6-27B 实测 MTP 接受率约 87/72/61%，平均每次接受 3-4 个 token，约 1.9x 速度提升。

---

*数据来源：Qwen 官方博客、DeepSeek 技术报告、Hugging Face 模型卡、Artificial Analysis、OpenLM.ai*
