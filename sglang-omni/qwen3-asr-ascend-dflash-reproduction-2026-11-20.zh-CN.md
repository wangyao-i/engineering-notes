# Qwen3-ASR-1.7B：昇腾 A3 DFlash 性能与复现记录（2026-11-20）

[English](qwen3-asr-ascend-dflash-reproduction-2026-11-20.en.md) | 更新：2026-11-20

本文整理自 [roadmap 原始测试报告](https://github.com/sgl-project/sglang-omni/issues/2076#issuecomment-7032927746)，记录 Qwen3-ASR-1.7B 在昇腾 A3 单 die 上的 DFlash 解码图配置、部署方法及七档并发结果。数据为测试执行方报告，本次文档整理没有重跑硬件测试。

本次测试使用 SGLang 0.5.23、SGLang-Omni 0.2.0、CANN 9.1.0 和驱动 26.0.rc1。原报告尚未给出镜像标签/digest、代码 commit、主模型/草稿 revision、640 条音频的精确清单和逐请求原始结果，因此下文是已报告配置的复现说明，不能保证按版本号即可逐值重现。安装环境参见 [NPU 安装指南](https://github.com/sgl-project/sglang-omni/blob/main/docs/other_platforms/installation_npu.md)，该指南随主线演进，当前推荐栈不代表本次实验栈。

---

## 1. 结论速览

| 项 | 值 |
|---|---|
| 配置 | SGLang-Omni **C64 + DFlash 投机解码** |
| 设备占用 | **单 die**（1 张卡的 1 个 die） |
| 本次最高吞吐 | **76.26 req/s**（并发 64） |
| 并发 64 延迟 | 平均 **825 ms**，P50 504 ms |
| 识别质量 | CER **0.0036**（中文，全档恒定） |
| 实时倍率 | 单并发 RTFx **60.1**，并发 64 RTFx **364.9** |

---

## 2. 已报告测试环境

### 2.1 硬件

| 项 | 测试环境 |
|---|---|
| 设备 | 昇腾 A3（Atlas 800I A3） |
| NPU 用量 | **1 个 die**（本文全部数据均在此条件测得） |
| 主机 CPU | 640 逻辑核（测试时避免其他主机负载） |
| 内存 | 2 TB（测试机容量，未证明为最低要求） |

### 2.2 软件

| 组件 | 版本 |
|---|---|
| 操作系统 | openEuler 22.03 LTS-SP4（aarch64） |
| 内核 | 5.10.0-216.0.0.115 |
| NPU 驱动 | 26.0.rc1（ascendhal 7.35.23） |
| CANN | 9.1.0 |
| Python | 3.12.x |
| torch | 2.10.0 |
| torch_npu | 2.10.0.post6 |
| **sglang** | **0.5.23** |
| **sglang-omni** | **0.2.0** |
| transformers | 5.19.0 |
| soundfile / librosa | 0.13.1 / 1.0.1 |
| jiwer / zhon | 4.0.0 / 2.0.2（压测 WER 依赖） |

> 本文所有数据在 `sglang 0.5.23 + sglang-omni 0.2.0 + CANN 9.1.0 + 驱动 26.0.rc1` 下测得。
> 其他版本组合不保证一致。

---

## 3. 模型准备

需要两个模型：

| 用途 | 模型 | 说明 |
|---|---|---|
| 主模型 | `Qwen/Qwen3-ASR-1.7B` | 待转写模型 |
| 草稿模型 | `Qwen3-ASR-1.7B-dflash` | **DFlash 投机解码必需**，是性能关键 |

```bash
export MODEL_DIR=/models/Qwen3-ASR-1.7B
export DRAFT_DIR=/models/Qwen3-ASR-1.7B-dflash

# 目录内容应包含 config.json / tokenizer* / *.safetensors 等
ls "$MODEL_DIR" "$DRAFT_DIR"
```

> 草稿模型需要与主模型及 DFlash 实现兼容。原报告仅提供本地模型目录，未固定草稿仓库/revision；不要将先前 PR 的草稿版本当成本次实验的已确认身份。主线支持范围和示例参见 [Qwen3-ASR cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/qwen3_asr.md#ascend-npu)。

---

## 4. 本次测试配置

保存为 `c64_dflash.yaml`：

```yaml
config_cls: Qwen3ASRPipelineConfig
name: qwen3-asr-c64
model_path: /models/Qwen3-ASR-1.7B
stages:
  asr:
    factory:
      dtype: bfloat16
      enable_async_decode: false
      enable_encoder_cuda_graph: false
    engine:
      max_running_requests: 64                 # 最大并发
      disable_cuda_graph: false
      cuda_graph_backend_decode: full          # 解码图全捕获
      cuda_graph_backend_prefill: disabled
      cuda_graph_max_bs: 64
      cuda_graph_bs: [1, 2, 4, 8, 16, 24, 32, 64]
      enable_torch_compile: false              # ★ 必须 false
      attention_backend: ascend
      speculative_algorithm: DFLASH            # ★ 性能关键
      speculative_draft_attention_backend: ascend
      speculative_draft_model_path: /models/Qwen3-ASR-1.7B-dflash
      speculative_num_draft_tokens: 8
```

### 关键项说明

| 配置 | 取值 | 作用 |
|---|---|---|
| `speculative_algorithm` | `DFLASH` | 启用 DFlash；本次没有关闭 DFlash 的对照，不能单独归因性能收益 |
| `speculative_num_draft_tokens` | `8` | 配置值为 8；实际 accepted/bonus 长度未在本次报告中给出 |
| `enable_torch_compile` | `false` | 保持本次支持配置：关闭编译；其他组合不在本次验证范围 |
| `cuda_graph_backend_decode` | `full` | 启用解码图执行 |
| `cuda_graph_bs` | 显式列出 8 个桶 | 图桶 [1,2,4,8,16,24,32,64]；包括全部测试档位 |
| `max_running_requests` | `64` | 与最高压测并发对齐 |
| `enable_async_decode` | `false` | 同步调度，与已合入的 NPU DFlash 支持范围一致 |

---

## 5. 部署步骤

### 5.1 启动服务

先准备具有上述运行时和 Omni 依赖的镜像，将下面的 `<sglang-omni-image>` 替换为实际镜像。原报告没有提供可直接拉取的精确镜像身份。先将 YAML 保存到宿主机 `/models/c64_dflash.yaml`，将音频和清单放到 `/data`；以下命令补充了 `/data` 挂载，并统一服务、单条请求和压测使用的模型名。`500g` 为原报告的共享内存设置，未证明是最低需求。设备映射和驱动路径按实际主机调整。

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
  <sglang-omni-image> \
  sleep infinity

# 进入容器启动服务
docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/c64_dflash.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'
```

> `ASCEND_RT_VISIBLE_DEVICES` 指定可见 die。原命令通过 privileged 模式暴露设备，应仅在受信主机使用；限制设备权限的部署方式见安装指南。

### 5.2 健康检查

```bash
curl -fsS http://127.0.0.1:8899/v1/models
```

检查响应包含所配置模型，再执行单条转写确认推理可用。原报告预计首次启动约需 2–4 分钟；实际耗时受加载和图捕获影响。

### 5.3 单条转写验证

```bash
curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
  -F file=@/data/audio/sample.wav \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=zh
```

### 5.4 接口说明

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

## 6. 压测方法

### 6.1 数据集格式

使用清单文件，每行以 `|` 分隔：

```
<样本ID>|<参考文本>|<音频绝对路径>|[可选附加字段]
```

示例：

```
000000|营造宁静和平的网络安全空间。|/data/audio/000000.wav
000001|今天天气很好。|/data/audio/000001.wav
```

> 原报告使用 Seed-TTS-Eval `zh` 的 640 条音频（24 kHz 单声道，均值 4.79 s），不是完整中文集。精确 revision、子集选择及清单哈希尚未提供；要对照表中数值，需使用同一清单。

### 6.2 压测命令

```bash
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 \
  --port 8899 \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --meta /data/meta.lst \
  --lang zh \
  --concurrencies 1,4,8,16,24,32,64 \
  --unique-audio \
  --warmup \
  --stream \
  --repeats 1 \
  --output result.json
```

### 6.3 参数说明

| 参数 | 说明 |
|---|---|
| `--concurrencies` | **逗号分隔字符串**，不是空格分隔多值 |
| `--unique-audio` | 按 benchmark 的音频去重规则选择样本，核对最终每档确为 640 条 |
| `--warmup` | 内置预热，**建议保留** |
| `--stream` | 流式请求，采集 TTFT 与 ITL；这些是文本响应指标，不是 NPU kernel 耗时 |
| `--lang zh` | 中文识别，启用中文 CER 归一化 |

### 6.4 预热

除 `--warmup` 外，建议正式测量前串行发 16 条请求使服务进入稳态：

```bash
head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done
```

> 将加载、捕获和预热从正式统计中分开；未预热可能把首次请求成本计入首档。保存服务日志以确认图捕获及实际执行模式。

### 6.5 测试期间的系统要求

| 要求 | 说明 |
|---|---|
| **独占 CPU** | 测试时主机上不要并行跑其他任务。本服务大量使用 CPU 线程做调度，CPU 争抢会显著抬高延迟 |
| **单 die 独占** | 每个 die 只跑一个实例 |
| 关闭其他负载 | 避免其他进程占用 NPU 显存或 PCIe 带宽 |

---

## 7. 实测性能数据

> 条件：昇腾 A3 **单 die**，独占主机 CPU，已充分预热，
> Seed-TTS-Eval `zh` 640 条（每档全部评测，640/640）。

### 7.1 主表

| 并发 | 平均延迟 | P50 | P95 | TTFT | 吞吐 | RTFx | CER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **79 ms** | 77 ms | 105 ms | 39.8 ms | **12.56 req/s** | 60.1 | 0.0036 |
| 4 | **260 ms** | 125 ms | 200 ms | 116.5 ms | **15.35 req/s** | 73.5 | 0.0036 |
| 8 | **254 ms** | 142 ms | 275 ms | 98.3 ms | **31.32 req/s** | 149.9 | 0.0036 |
| 16 | **259 ms** | 177 ms | 306 ms | 98.2 ms | **61.35 req/s** | 293.6 | 0.0036 |
| 24 | **404 ms** | 275 ms | 518 ms | 171.3 ms | **58.74 req/s** | 281.1 | 0.0036 |
| 32 | **442 ms** | 323 ms | 1188 ms | 224.3 ms | **71.62 req/s** | 342.7 | 0.0036 |
| 64 | **825 ms** | 504 ms | 3622 ms | 245.0 ms | **76.26 req/s** | 364.9 | 0.0036 |

### 7.2 实时率

| 并发 | RTF（平均） | ITL（平均） |
|---:|---:|---:|
| 1 | 0.0168 | 14.0 ms |
| 4 | 0.0547 | 44.4 ms |
| 8 | 0.0540 | 49.2 ms |
| 16 | 0.0524 | 49.3 ms |
| 24 | 0.0831 | 62.7 ms |
| 32 | 0.0929 | 57.7 ms |
| 64 | 0.1726 | 123.2 ms |

> 每请求 RTF = 请求耗时 / 音频时长，表中为其平均值。RTFx = 完成音频总时长 / 整轮墙钟时间，反映聚合处理能力；并发情况下它不是平均 RTF 的倒数。TTFT/ITL 为流式文本响应指标。

### 7.3 关键特征

| 观察 | 数据 |
|---|---|
| 本次吞吐范围 | 12.56 → 76.26 req/s（并发 1 → 64），约 **6.1×**；并发 16 → 24 时从 61.35 降到 58.74 req/s，非单调增长 |
| 平均延迟变化 | 79 → 825 ms，约 **10.4×**；选择并发需同时评估尾延迟 |
| 本次 CER | 七档均报告 **0.0036（0.36%）**；本次没有无 DFlash 对照，不能据此宣称无损 |
| 尾延迟 | P95 在并发 ≤16 时 ≤306 ms；并发 64 时 3622 ms |

> 并发 64 的吞吐为本次最高，但 P95 为 3622 ms；本报告没有给出尾延迟归因证据。并发 ≤16 时本次 P95 ≤306 ms，并发 32 时为 1188 ms。按业务延迟目标选档，不将本次 C64 结果视为通用最优。

---

## 8. 复现检查清单

| # | 检查项 | 不满足的后果 |
|---|--------|-------------|
| 1 | 主模型与**草稿模型**均已就位 | DFlash 无法启动 |
| 2 | 配置文件里 `speculative_algorithm: DFLASH` | 与本次测试配置不一致 |
| 3 | `enable_torch_compile: false` | 超出本次验证配置 |
| 4 | 保留图桶，并检查捕获和执行日志 | 不能确认运行了预期图路径 |
| 5 | 按运行时需要设置共享内存 | 共享内存不足可能影响启动或运行 |
| 6 | `ASCEND_RT_VISIBLE_DEVICES` 指向**空闲 die** | 显存不足或启动失败 |
| 7 | 测试前执行报告中的预热流程 | 首次捕获等成本可能混入正式测量 |
| 8 | 测试期间主机无其他负载 | 延迟显著抬升（CPU 争抢） |
| 9 | `--concurrencies` 用逗号分隔 | 参数解析失败 |
| 10 | 已安装 benchmark 所需依赖，包括 `jiwer` + `zhon` | 精度评估或脚本导入可能失败 |

---

## 9. 已知限制

| 项 | 说明 |
|---|---|
| 单卡线性度 | 本文仅测单 die；多 die 并行的扩展性未覆盖 |
| 音频长度 | 数据集均值 4.79 s，长音频（>30 s）表现未覆盖 |
| 语言 | 仅测中文（`zh`）；其他语种需重测 |
| 并发上限 | 压测最高 64；更高并发的行为未验证 |
| 显存占用 | 未记录峰值；`mem_fraction_static` 使用默认值 |\n| 重复性 | 每档一次正式测量，没有跨轮离散度；精确镜像、代码/权重 revision、清单及原始结果待补齐 |

---

## 10. 最小复现命令汇总

以下命令在前述镜像、模型、YAML、音频及清单均已准备后执行。benchmark 所在的 Python 环境需能访问服务和 `/data`。

```bash
# 1. 启动
docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/c64_dflash.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'

# 2. 等待就绪
until curl -fsS -o /dev/null http://127.0.0.1:8899/v1/models; do sleep 5; done

# 3. 预热
head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done

# 4. 压测
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port 8899 --model-path Qwen/Qwen3-ASR-1.7B --meta /data/meta.lst --lang zh \
  --concurrencies 1,4,8,16,24,32,64 \
  --unique-audio --warmup --stream --repeats 1 --output result.json
```

