# Qwen3-ASR-1.7B 昇腾 A3 性能调优与复现指南（2026-11-23）

[English](qwen3-asr-ascend-tuning-results-2026-11-23.en.md) | 更新：2026-11-23

本文整理自 [roadmap 中的完整调优记录](https://github.com/sgl-project/sglang-omni/issues/2076#issuecomment-7095348788)，记录 Qwen3-ASR-1.7B 在昇腾 A3 **单 die** 上从初始基线 B0 逐项叠加到 B2b / B3 的调优过程、方法、配置、部署、压测与实测数据，供其他开发者按步骤复现。数据为测试执行方报告，本次文档整理没有重跑硬件测试。

本次实验使用 SGLang 0.5.23、SGLang-Omni 0.2.0、CANN 9.1.0、驱动 26.0.rc1。原报告尚未固定镜像标签/digest、代码 commit、主模型/草稿 revision、640 条音频的精确清单与逐请求原始结果，因此下文是**已报告配置的复现说明**，不能保证仅凭版本号即可逐值重现。环境安装参见 [NPU 安装指南](https://github.com/sgl-project/sglang-omni/blob/main/docs/other_platforms/installation_npu.md)；该指南随主线演进，其当前推荐栈不代表本次实验栈。

> **口径提示**：本文是 **B0/B2b/B3 三配置调优实验**（并发 1/16/32/64，A-B-B-A 交叉，默认**含重复音频**口径）。它与 [2026-11-20 七档 DFlash 单次记录](https://github.com/wangyao-i/engineering-notes/blob/991b61614f1ccc4d824a6a241d8f5cec1a7d61c6/sglang-omni/qwen3-asr-ascend-dflash-reproduction-2026-11-20.zh-CN.md)（并发 1/4/8/16/24/32/64，`--unique-audio` 去重口径）是**不同实验**，workload、配置与去重口径均不同，数值不可直接互比。
>
> **阅读顺序建议**：§1 结论 → §5 调优方法 → §4 配置 → §6 部署 → §7 压测 → §8 数据 → §9 稳定性。

---

## 1. 结论速览：两套推荐配置

调优得到两个不同取向的配置，按目标二选一。所有加速比均为 **A-B-B-A 交叉验证值**，不是单段值（见 §9）。

### 1.1 吞吐优先 —— 配置 B2b

| 项 | 值 |
|---|---|
| 内容 | 初始基线 + 专用编码器服务 + prefill 图 + **关闭异步解码**（不加 DFlash） |
| 吞吐（并发 64） | 摘要 **87 – 119 req/s**；单次结果表 **122.10 req/s** |
| 延迟（并发 64） | **526 – 716 ms**（A-B-B-A 两段） |
| 单并发延迟 | 114 ms |
| CER | 摘要 c64 0.0036 – 0.0038；单次表 c64 0.0040；c1 0.0248 |
| 相对初始基线 | **快 3.29×**（A-B-B-A：2042 → 621 ms） |

### 1.2 质量优先 —— 配置 B3

| 项 | 值 |
|---|---|
| 内容 | B2b + DFlash 投机解码（draft 8） |
| 吞吐（并发 64） | 摘要约 **65 req/s**；单次结果表 **81.56 req/s** |
| 延迟（并发 64） | **958 – 1628 ms**（A-B-B-A 均值 1293 ms）；单次表 769 ms |
| 单并发延迟 | **62 ms** |
| CER | **报告各档恒定 0.0036** |
| 相对初始基线 | 1.40×（A-B-B-A 交叉验证） |

### 1.3 怎么选

| 场景 | 选择 | 理由 |
|---|---|---|
| 高并发批量转写 | **B2b** | 吞吐显著更高（单次表 c32 为 B3 的 2.4×），c64 CER 仅差 0.0004 |
| 低并发 / 精度敏感 | **B3** | c1 延迟快 1.84×（114 → 62 ms），c1 CER 好 6.9×（0.0248 → 0.0036） |
| 需要对外承诺延迟 | **两者都需谨慎** | B3 段间离散 51.8%、B2b 30.6%，见 §9 |

> **摘要口径与单次表口径不同**：摘要吞吐范围（B2b 87–119、B3 约 65 req/s）与单次结果表（122.10、81.56 req/s）来自不同测量口径，本文分开保留，不合并为单一点估计。CER 同理（摘要 c64 0.0036–0.0038 vs 单次表 0.0040）。

---

## 2. 测试环境

### 2.1 硬件

| 项 | 测试环境 |
|---|---|
| 设备 | 昇腾 A3（Atlas 800I A3） |
| NPU 用量 | **1 个 die**（本文全部数据均在此条件测得） |
| 主机 CPU | 测试期间**独占**，避免其他主机负载（本服务大量使用 CPU 线程做调度） |
| 其他主机规格 | 核数/内存未在本实验单独记录 |

### 2.2 软件

| 组件 | 版本 |
|---|---|
| 操作系统 | openEuler 22.03 LTS-SP4（aarch64） |
| 内核 | 5.10.0-216.0.0.115 及以上 |
| NPU 驱动 | 26.0.rc1（ascendhal 7.35.23） |
| CANN | 9.1.0 |
| Python | 3.12.x |
| torch / torch_npu | 2.10.0 / 2.10.0.post6 |
| **sglang** | **0.5.23** |
| **sglang-omni** | **0.2.0** |
| transformers | 5.19.0 |
| soundfile / librosa | 0.13.1 / 1.0.1 |
| **jiwer / zhon** | **4.0.0 / 2.0.2**（压测精度评估必需） |

> 本文所有数据在 `sglang 0.5.23 + sglang-omni 0.2.0 + CANN 9.1.0 + 驱动 26.0.rc1` 下测得，其他版本组合不保证一致。
>
> **jiwer 与 zhon 不在默认镜像内**，缺失会导致压测以 `ModuleNotFoundError: No module named 'jiwer'` 直接退出（部署步骤 §6 已包含安装命令）。

---

## 3. 模型准备

| 用途 | 模型 | 说明 |
|---|---|---|
| 主模型 | `Qwen/Qwen3-ASR-1.7B` | 待转写模型，B0/B2b/B3 均需 |
| 草稿模型 | `Qwen3-ASR-1.7B-dflash` | **仅 B3 需要**，DFlash 投机解码必需 |

```bash
export MODEL_DIR=/models/Qwen3-ASR-1.7B
export DRAFT_DIR=/models/Qwen3-ASR-1.7B-dflash

# 目录应包含 config.json / tokenizer* / *.safetensors 等
ls "$MODEL_DIR" "$DRAFT_DIR"
```

> 草稿模型必须与主模型及 DFlash 实现配套，否则投机接受率显著下降。有效上下文应保持在草稿模型的 **32768 token** 注意力窗口内。原报告仅提供本地模型目录，未固定草稿仓库/revision；不要把先前 PR 的草稿版本当成本次实验的已确认身份。主线支持范围与示例参见 [Qwen3-ASR cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/qwen3_asr.md#ascend-npu)。

---

## 4. 配置文件

三份配置均为完整可复制 YAML。B2b / B3 在 B0 基础上逐项叠加（叠加逻辑与归因见 §5）。`<MODEL_DIR>` / `<DRAFT_DIR>` 替换为实际路径。

### 4.1 初始基线 B0（调优起点）

保存为 `b0.yaml`。这是全部叠加实验的起点，也是常见默认写法。

```yaml
config_cls: Qwen3ASRPipelineConfig
name: baseline
model_path: <MODEL_DIR>
stages:
  asr:
    factory:
      dtype: bfloat16
      enable_encoder_cuda_graph: false
      enable_pre_lm_encoder: false        # 未启用专用编码器服务
    engine:
      max_running_requests: 64
      disable_cuda_graph: false
      cuda_graph_backend_decode: full
      cuda_graph_backend_prefill: disabled  # 未启用 prefill 图
      cuda_graph_max_bs: 64
      cuda_graph_bs: [1, 2, 4, 8, 16, 24, 32, 64]
      enable_torch_compile: false
      attention_backend: ascend
      mem_fraction_static: 0.9
      context_length: 32768
      page_size: 128
      chunked_prefill_size: 4096
      mm_attention_backend: ascend_attn
      # 无 speculative_* 段（即无 DFlash）
    tp_size: 1
```

> 基线实测：c1 128 ms / c64 2094 ms，c64 吞吐 30.20 req/s，CER 0.0036–0.0226。

### 4.2 配置 B2b（吞吐优先）

保存为 `b2b.yaml`。相对 B0 改动 `factory` 与 `engine` 各一处（★ 标注）。

```yaml
config_cls: Qwen3ASRPipelineConfig
name: b2b
model_path: <MODEL_DIR>
stages:
  asr:
    factory:
      dtype: bfloat16
      enable_encoder_cuda_graph: false
      enable_async_decode: false          # ★ 关键：单独带来 1.85×
      enable_pre_lm_encoder: true         # ★ 关键：单独带来 2.64×
      pre_lm_max_batch_size: 16
      pre_lm_max_batch_wait_ms: 2
      pre_lm_cache_max_entries: 1024
      pre_lm_cache_size_bytes: 2147483648
    engine:
      max_running_requests: 64
      disable_cuda_graph: false
      cuda_graph_backend_decode: full
      cuda_graph_backend_prefill: breakable   # ★ 仅 1.02×，可选
      cuda_graph_max_bs: 64
      cuda_graph_bs: [1, 2, 4, 8, 16, 24, 32, 64]
      enable_torch_compile: false
      attention_backend: ascend
      mem_fraction_static: 0.9
      context_length: 32768
      page_size: 128
      chunked_prefill_size: 4096
      mm_attention_backend: ascend_attn
    tp_size: 1
```

### 4.3 配置 B3（质量优先）

保存为 `b3.yaml`。在 B2b 基础上追加投机解码段（★ 标注），需要草稿模型。

```yaml
config_cls: Qwen3ASRPipelineConfig
name: b3
model_path: <MODEL_DIR>
stages:
  asr:
    factory:
      dtype: bfloat16
      enable_encoder_cuda_graph: false
      enable_async_decode: false
      enable_pre_lm_encoder: true
      pre_lm_max_batch_size: 16
      pre_lm_max_batch_wait_ms: 2
      pre_lm_cache_max_entries: 1024
      pre_lm_cache_size_bytes: 2147483648
    engine:
      max_running_requests: 64
      disable_cuda_graph: false
      cuda_graph_backend_decode: full
      cuda_graph_backend_prefill: breakable
      cuda_graph_max_bs: 64
      cuda_graph_bs: [1, 2, 4, 8, 16, 24, 32, 64]
      enable_torch_compile: false
      attention_backend: ascend
      mem_fraction_static: 0.9
      context_length: 32768
      page_size: 128
      chunked_prefill_size: 4096
      mm_attention_backend: ascend_attn
      speculative_algorithm: DFLASH           # ★ CER 0.0248 → 0.0036
      speculative_draft_attention_backend: ascend
      speculative_draft_model_path: <DRAFT_DIR>
      speculative_num_draft_tokens: 8         # ★ 不要设 16，实测更慢
    tp_size: 1
```

### 4.4 参数层级（放错会启动失败）

| 参数 | 正确层级 | 放错时报错 |
|---|---|---|
| `enable_async_decode` | `factory:` | `TypeError: Unexpected keyword argument` |
| `enable_pre_lm_encoder` / `pre_lm_*` | `factory:` | 同上 |
| `request_build_*` / `prefill_coalesce_*` | `factory:` | 同上 |
| `speculative_*` / `cuda_graph_*` / `page_size` / `mem_fraction_static` | `engine:` | — |

> 原因：`engine:` 下的字段会被直接透传给底层 `ServerArgs(**kwargs)`，而 `enable_async_decode` 等属于 factory 层参数，不在 `ServerArgs` 里。

---

## 5. 调优方法（本文最关键的一节）

### 5.1 四条必须遵守的规则

| # | 规则 | 违反的后果（实测） |
|---|---|---|
| 1 | **从初始基线出发，逐项叠加，一次只改一项** | 若从「已含 DFlash 的配置」起步，DFlash 的既有效果会被算进后续项，归因全错 |
| 2 | **同时改两项会互相掩盖** | 「加 DFlash」与「关异步解码」一起改，两者方向相反、几乎抵消，会被误判为「只有 1.01×」 |
| 3 | **必须做 A-B-B-A 交叉或 ≥2 段重复** | 单段测得的加速比可高估 **1.9 倍**（实测 2.72× → 1.40×） |
| 4 | **每档样本 ≥320，且固定每档请求数** | 样本过少曲线非单调，结论不可信 |

### 5.2 阶梯设计

| 代号 | 改动 | 目的 |
|---|---|---|
| **B0** | 初始基线 | 参照点 |
| **B1** | + 专用编码器服务（`enable_pre_lm_encoder: true` + `pre_lm_*`） | 隔离编码器服务 |
| **B2** | + prefill CUDA 图（`breakable`） | 隔离 prefill 图 |
| **B2b** | + `enable_async_decode: false` | 隔离异步解码 |
| **B3** | + DFlash（draft 8） | 隔离投机解码 |

> 注意 B2b：这一步**不加 DFlash**，专门用来把「关异步解码」与「加 DFlash」拆开。少了这一步就会得出错误结论。

### 5.3 A-B-B-A 交叉怎么跑

```
段1: B0   ← A
段2: 候选 ← B
段3: 候选 ← B
段4: B0   ← A
```

每段落都要：**重启服务 → 串行预热 16 条 → 压测 → 记录**。最后比较：

```
候选均值  = (段2 + 段3) / 2
基线均值  = (段1 + 段4) / 2
加速比    = 基线均值 / 候选均值
段间离散  = |段2 - 段3| / 候选均值 × 100%
```

**段间离散本身就是结论的一部分**：离散 >30% 说明该配置不适合作为承诺值。

### 5.4 调优过程记录（单项归因）

| 步骤 | c64 延迟 | 倍数 | 备注 |
|---|---:|---:|---|
| B0 初始基线 | 2094 ms | — | 参照 |
| B1 + 编码器服务 | 792 ms | **2.64×** | 最大的单项收益 |
| B2 + prefill 图 | 774 ms | 1.02× | 收益极小 |
| B2b + 关异步解码 | 508 ms | **1.85×** | 第二个大杠杆 |
| B3 + DFlash | ~1293 ms | 0.39×（变慢） | 换来精度，见 §8.3 |

> 上表为逐项叠加的单段观测，用于归因；对外报告的加速比以 §9 的 A-B-B-A 交叉值为准。

---

## 6. 部署

### 6.1 启动容器与服务

先准备具有上述运行时与 Omni 依赖的镜像，将 `<sglang-omni-image>` 替换为实际镜像（原报告未提供可直接拉取的精确镜像身份）。将 YAML 保存到宿主机 `/models/`，音频与清单放到 `/data`。`--shm-size=500g` 为原报告设置（未证明为最低需求），但**默认 64 MB 会导致启动失败**，必须放大。设备映射与驱动路径按实际主机调整。

```bash
docker run -d --name sglang-omni-asr \
  --privileged=true --net=host --shm-size=500g \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /models:/models \
  -v /data:/data \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  <sglang-omni-image> sleep infinity

# 压测精度依赖（默认镜像缺失）
docker exec sglang-omni-asr bash -lc 'python -m pip install jiwer zhon'

# 启动服务（按目标配置选择 b0/b2b/b3.yaml）
docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/b2b.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'
```

> `ASCEND_RT_VISIBLE_DEVICES` 指定可见 die。privileged 模式暴露设备，应仅在受信主机使用；限制设备权限的部署方式见安装指南。
>
> **模型名必须一致**：服务 `--model-name`、单条请求与压测使用的模型名需统一（本文统一为 `Qwen/Qwen3-ASR-1.7B`）。

### 6.2 健康检查与单条转写

```bash
# 就绪检查（首次启动约 2–4 分钟，受加载与图捕获影响）
curl -fsS http://127.0.0.1:8899/v1/models

# 单条转写验证推理可用
curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
  -F file=@/data/audio/sample.wav \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=zh
```

### 6.3 接口说明

| 项 | 值 |
|---|---|
| 转写接口 | `POST /v1/audio/transcriptions` |
| 请求格式 | `multipart/form-data`，音频放在 `file` 字段 |
| 关键字段 | `model`、`language`、`stream` |
| 流式响应 | SSE，事件为 `transcript.text.delta` / `transcript.text.done` |

流式响应示例：

```
data: {"type": "transcript.text.delta", "delta": "营"}
data: {"type": "transcript.text.delta", "delta": "造"}
data: {"type": "transcript.text.done", "text": "营造宁静和平的网络安全空间。"}
data: [DONE]
```

---

## 7. 压测方法

### 7.1 数据集与口径

清单文件每行以 `|` 分隔：`<样本ID>|<参考文本>|<音频绝对路径>`

```
000000|营造宁静和平的网络安全空间。|/data/audio/000000.wav
000001|今天天气很好。|/data/audio/000001.wav
```

> 本文使用公开数据集 **Seed-TTS-Eval `zh`**（24 kHz 单声道，均值 4.79 s）。
> **该数据集同一参考音频配多个目标文本，行数远大于唯一音频数**（640 行仅 320 唯一音频）。
> 是否去重会显著改变结果，**报告时必须注明是否加了 `--unique-audio`**。
> **本文为默认含重复口径（不加 `--unique-audio`）**，每档报告 640 条请求。这与 2026-11-20 七档记录的去重口径不同。
> 精确 revision、子集选择与清单哈希尚未提供；要对照表中数值，需使用同一清单。

### 7.2 压测命令

在能访问服务与 `/data`、且含匹配 Omni checkout 与压测依赖的 Python 环境中执行：

```bash
docker exec sglang-omni-asr bash -lc "cd <SGLANG_OMNI_REPO> && python -u \
  -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port 8899 --model-path Qwen/Qwen3-ASR-1.7B \
  --meta /data/meta.lst --lang zh \
  --concurrencies 1,16,32,64 --warmup --stream --repeats 1 \
  --output result.json"
```

| 参数 | 说明 |
|---|---|
| `--concurrencies` | **逗号分隔字符串**（空格分隔会报 `unrecognized arguments`）；本文档位 1,16,32,64 |
| `--warmup` | 内置预热，建议保留 |
| `--stream` | 流式请求，采集 TTFT 与 ITL（文本响应指标，非 NPU kernel 耗时） |
| `--unique-audio` | 去重，会改变口径；**本文不加**，若加必须标注 |
| `--repeats` | 单次表用 1；**稳定性测量建议 ≥2 取中位数，或做 A-B-B-A**（本配置失稳风险高） |
| `--max-samples` | **不建议低于 320**，否则曲线非单调 |

### 7.3 额外预热（必须做）

除 `--warmup` 外，正式测量前串行发 16 条请求使服务进入稳态：

```bash
head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done
```

> 未预热会使首档数据虚高数倍（计算图捕获开销被计入首个请求）。保存服务日志以确认图捕获与实际执行模式。

### 7.4 测试期间要求

| 要求 | 说明 |
|---|---|
| **独占 CPU** | 服务大量使用 CPU 线程做调度，CPU 争抢会显著抬高延迟 |
| **单 die 独占** | 每个 die 只跑一个实例 |
| 关闭其他负载 | 避免其他进程占用 NPU 显存或 PCIe 带宽 |
| 记录 NPU 温度/频率 | 用于排查会话间漂移 |
| 多段重复 | 见 §5.3 A-B-B-A |

---

## 8. 实测数据

> 条件：昇腾 A3 **单 die**，独占主机 CPU，已充分预热，Seed-TTS-Eval `zh` 默认含重复口径，每档 640 条请求。以下分档表为**单次结果**；摘要范围与交叉值另见 §1、§9。

### 8.1 三配置延迟与精度（单次）

| 并发 | B0 初始基线〔延迟 / CER〕 | **B2b（吞吐优先）**〔延迟 / CER〕 | **B3（质量优先）**〔延迟 / CER〕 |
|---:|---|---|---|
| 1 | 128 ms / 0.0226 | **114 ms / 0.0248** | 62 ms / 0.0036 |
| 16 | 662 ms / 0.0067 | **332 ms / 0.0119** | 450 ms / 0.0036 |
| 32 | 1404 ms / 0.0075 | **494 ms / 0.0048** | 1139 ms / 0.0036 |
| 64 | 2094 ms / 0.0036 | **508 ms / 0.0040** | 769 ms / 0.0036 |

> B3 的 c64 单次为 769 ms；A-B-B-A 两段为 958–1628 ms（均值 1293 ms），见 §9。

### 8.2 吞吐（req/s，单次）

| 并发 | B0 | **B2b** | **B3** |
|---:|---:|---:|---:|
| 1 | 7.80 | **8.79** | 15.96 |
| 16 | 23.96 | **47.80** | 35.37 |
| 32 | 22.62 | **63.67** | 26.43 |
| 64 | 30.20 | **122.10** | 81.56 |

> B3 吞吐取单段值（其延迟失稳，吞吐亦随之波动）。摘要中 B2b 87–119、B3 约 65 req/s 与本表口径不同，分开保留。

### 8.3 DFlash 的取舍（重要）

| 并发 | 无 DFlash（B2b） | 有 DFlash（B3） | 结论 |
|---:|---|---|---|
| 1 | 114 ms / CER 0.0248 | **62 ms / CER 0.0036** | DFlash 全面胜出 |
| 16 | **332 ms / CER 0.0119** | 450 ms / CER 0.0036 | 延迟 DFlash 输，精度赢 |
| 64 | **508 ms / CER 0.0040** | 1293 ms / CER 0.0036 | 精度只差 0.0004，延迟差约 2.5× |

**机理（为什么 DFlash 高并发不提速）**：投机解码是用算力换带宽。8 个草稿 token 一次验证要算 **9 个位置**，按接受长度 5.498 计算，**每 token 的算力约为原来的 1.64×**（9 / 5.498）。

| 场景 | 瓶颈 | DFlash 效果 |
|---|---|---|
| 低并发（c1） | 内存带宽 | **有效**（1.84×） |
| 高并发（c16+） | 算力饱和 | **失效甚至变慢** |

> 接受长度 5.498 来自 roadmap 引用的历史 eager/graph A-B-B-A 报告（含约 4.498 个被接受提议 + 1 个 bonus，接受率 64.2%）；本次三配置实验未单独观测接受率。

### 8.4 CER 归因（已隔离验证）

| 配置 | 异步解码 | DFlash | c1 CER | c16 CER | c32 CER | c64 CER |
|---|---|---|---:|---:|---:|---:|
| B2 | 开 | 无 | 0.0248 | 0.0155 | 0.0040 | 0.0044 |
| **B2b** | **关** | 无 | **0.0248** | **0.0119** | 0.0048 | 0.0040 |
| **B3** | 关 | **有** | — | — | **0.0036** | **0.0036** |

**B2 与 B2b 在 c1 的 CER 完全相同（0.0248）→ 关闭异步解码不影响精度。** 加入 DFlash 后才降到 0.0036，**CER 改善确实来自 DFlash**。

---

## 9. 稳定性与交叉验证（必须报告）

A-B-B-A 交叉测量结果（每配置两段）：

| 配置 | c32 两段 (ms) | c64 两段 (ms) | c64 段间离散 |
|---|---|---|---:|
| **B0 初始基线** | 1205 / 1058 | 1984 / 2100 | **5.7%（最稳定）** |
| **B2b（吞吐优先）** | 545 / 499 | 716 / 526 | **30.6%** |
| **B3（质量优先）** | 956 / 1222 | 958 / 1628 | **51.8%（最不稳定）** |

对应加速比（**交叉验证值，不是单段值**）：

| 配置 | c32 (×) | c64 (×) |
|---|---:|---:|
| B2b vs B0 | 2.17× | **3.29×** |
| B3 vs B0 | 1.07× | 1.40× |

**单段值与交叉值差异（说明为什么必须做交叉）**：

| 配置 | 单段测得的加速比 | 交叉验证值 | 高估幅度 |
|---|---:|---:|---:|
| B2b | 4.12× | **3.29×** | 1.25× |
| B3 | 2.72× | **1.40×** | **1.94×** |

> 段间离散 >30% 的配置（B2b、B3）不适合作为对外承诺值；B0 仅 5.7%。选择配置时应同时考虑吞吐、精度、尾延迟与复现稳定性。

---

## 10. 无效与有害的配置（避免重复踩坑）

| 尝试 | 结果 | 说明 |
|---|---|---|
| `page_size: 64` | ❌ **CER 崩到 8+** | 输出几乎全错，**绝不可用** |
| `speculative_num_draft_tokens: 16` | ❌ c64 由 769 → 1137 ms（0.68×） | 保持 8 |
| `max_running_requests: 80` | ❌ 启动失败 | `cuda_graph_max_bs must cover max_running_requests (64 < 80)` |
| `prefill_coalesce_*` 调整 | ⚠️ 无稳定收益 | 方向不一致，属噪声 |
| `request_build_max_workers` 8/16/32 | ⚠️ 无改善 | 默认 1，调大反而更差 |
| `mem_fraction_static: 0.92` | ⚠️ 噪声内 | 不推荐 |
| `enable_torch_compile: true` | ❌ 解码图捕获失败 | 必须 false |

### 10.1 两个额外约束

**① `max_running_requests` 受 CUDA 图桶覆盖约束**：要提高并发，必须同步提高 `cuda_graph_max_bs` 并扩 `cuda_graph_bs` 桶，否则启动即报错（见上表）。

**② `prefill_coalesce_wait_ms` 必须 > 0**：

```yaml
prefill_coalesce_requests: 0        # 关闭合并
prefill_coalesce_wait_ms: 0.1       # ✅ 不能写 0.0（schema 要求 gt=0）
```

---

## 11. 复现检查清单

| # | 检查项 | 不满足的后果 |
|---|---|---|
| 1 | 主模型就位；（B3）草稿模型也需就位 | DFlash 无法启动 |
| 2 | **`enable_async_decode: false` 放在 `factory:`** | 吞吐少 1.85×；放 engine 下报错 |
| 3 | **`enable_pre_lm_encoder: true` + `pre_lm_*` 在 `factory:`** | 吞吐少 2.64× |
| 4 | `speculative_num_draft_tokens: 8` | 设 16 会慢 1.48× |
| 5 | **`page_size` 保持 128** | 设 64 会让 CER 崩到 8+ |
| 6 | `enable_torch_compile: false` | 解码图捕获失败 |
| 7 | `cuda_graph_bs` 覆盖目标最大并发 | 高并发退化为 eager |
| 8 | `--shm-size` 已放大 | 启动失败 |
| 9 | 已装 `jiwer` + `zhon` | 压测直接退出 |
| 10 | `--concurrencies` 用逗号分隔 | 参数解析失败 |
| 11 | 已完成预热（16 条串行 + `--warmup`） | 首档虚高数倍 |
| 12 | 测试期间主机无其他负载、单 die 独占 | 延迟显著抬升 |
| 13 | **已做 A-B-B-A 或多段重复** | 单段可高估 1.9× |
| 14 | 已记录是否 `--unique-audio`（本文默认不加） | 口径不可比 |
| 15 | 每档样本 ≥320 | 曲线非单调 |

---

## 12. 已知限制

| 项 | 说明 |
|---|---|
| **超出上游已验证范围** | 上游仅在 decode 桶 [1,2]、encoder/prefill 图禁用的条件下验证过；本文启用 prefill 图并扩桶到 64，**正确性需自行回归** |
| 候选配置均有漂移 | B2b 段间 30.6%、B3 段间 51.8%；初始基线仅 5.7% |
| 单 die | 未测多 die 横向扩展，不能线性外推到整机 |
| 无投机接受率观测 | 本次拿不到接受率，无法进一步调优 draft 数（§8.3 的 5.498 引自历史报告） |
| 样本量 | 每档 640 请求；交叉验证仅 2 段/配置 |
| 音频长度 | 均值 4.79 s，>30 s 未覆盖 |
| 语言 | 仅中文 |
| 并发上限 | 最高 64 |
| 显存峰值 | 未记录 |
| 复现身份 | 精确镜像、代码/模型 revision、清单与逐请求原始结果待补齐 |

---

## 13. 最小复现命令汇总

以下命令假设镜像、模型、YAML、音频与清单均已按前述准备。压测环境需能访问服务与 `/data`。

```bash
# 1. 启动（按目标配置选择 b0/b2b/b3.yaml）
docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/b2b.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'

# 2. 等待就绪
until curl -fsS -o /dev/null http://127.0.0.1:8899/v1/models; do sleep 5; done

# 3. 预热（16 条串行）
head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done

# 4. 压测（单次表口径：含重复音频，不加 --unique-audio）
docker exec sglang-omni-asr bash -lc "cd <SGLANG_OMNI_REPO> && python -u \
  -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port 8899 --model-path Qwen/Qwen3-ASR-1.7B \
  --meta /data/meta.lst --lang zh \
  --concurrencies 1,16,32,64 --warmup --stream --repeats 1 --output result.json"
```

> **复现加速比与离散度**：按 §5.3 跑 A-B-B-A（B0 → 候选 → 候选 → B0），每段重启服务并重新预热，再按公式计算交叉加速比与段间离散。仅跑单段会高估加速比（B3 可高估 1.94×）。

---

## 14. 一页速查

```yaml
# 吞吐优先（B2b）相对 B0 的最小改动集
factory:
  enable_async_decode: false          # +1.85×
  enable_pre_lm_encoder: true         # +2.64×
  pre_lm_max_batch_size: 16
  pre_lm_max_batch_wait_ms: 2
  pre_lm_cache_max_entries: 1024
  pre_lm_cache_size_bytes: 2147483648
engine:
  cuda_graph_backend_prefill: breakable   # +1.02×

# 质量优先（B3）= B2b 再加：
engine:
  speculative_algorithm: DFLASH            # CER → 0.0036
  speculative_draft_attention_backend: ascend
  speculative_draft_model_path: <DRAFT_DIR>
  speculative_num_draft_tokens: 8
```

**不要动**：`page_size`（保持 128）、`enable_torch_compile`（保持 false）、`speculative_num_draft_tokens`（保持 8）。