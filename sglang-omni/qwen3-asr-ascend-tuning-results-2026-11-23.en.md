# Qwen3-ASR-1.7B on Ascend A3: Tuning and Reproduction Guide (2026-11-23)

[中文版](qwen3-asr-ascend-tuning-results-2026-11-23.zh-CN.md) | Updated: 2026-11-23

This document is adapted from the [complete tuning record in the roadmap](https://github.com/sgl-project/sglang-omni/issues/2076#issuecomment-7095348788). It records the step-by-step tuning of Qwen3-ASR-1.7B on one Ascend A3 die, from the initial baseline B0 through B2b / B3: methodology, configuration, deployment, benchmarking and measured data, so that other developers can reproduce it. Results were reported by the test operator; preparing this document did not involve another hardware run.

The experiment used SGLang 0.5.23, SGLang-Omni 0.2.0, CANN 9.1.0 and driver 26.0.rc1. The original report does not pin the image tag/digest, source commit, target/draft revisions, the exact 640-clip manifest or raw per-request results. This is therefore a **reproduction procedure for the reported configuration**, not a guarantee of identical measurements from package versions alone. See the [NPU installation guide](https://github.com/sgl-project/sglang-omni/blob/main/docs/other_platforms/installation_npu.md) for environment setup; its current mainline recommendations do not identify this experiment's stack.

> **Scope note**: this is the **B0/B2b/B3 three-profile tuning experiment** (concurrency 1/16/32/64, A-B-B-A cross-validation, default **with-duplicate-audio** scope). It is a **different experiment** from the [2026-11-20 seven-level DFlash single-run record](https://github.com/wangyao-i/engineering-notes/blob/991b61614f1ccc4d824a6a241d8f5cec1a7d61c6/sglang-omni/qwen3-asr-ascend-dflash-reproduction-2026-11-20.en.md) (concurrency 1/4/8/16/24/32/64, `--unique-audio` deduplicated scope). Workload, configuration and deduplication scope differ; the numbers are not directly comparable.
>
> **Suggested reading order**: §1 conclusions → §5 tuning method → §4 configuration → §6 deployment → §7 benchmark → §8 data → §9 stability.

---

## 1. Conclusions at a glance: two recommended profiles

Tuning produced two profiles with different priorities; choose one by goal. All speedups below are **A-B-B-A cross-validated values**, not single-segment values (see §9).

### 1.1 Throughput priority — profile B2b

| Item | Value |
|---|---|
| Content | Initial baseline + dedicated encoder service + prefill graph + **async decode off** (no DFlash) |
| Throughput (concurrency 64) | Headline **87 – 119 req/s**; single-run table **122.10 req/s** |
| Latency (concurrency 64) | **526 – 716 ms** (two A-B-B-A segments) |
| Single-concurrency latency | 114 ms |
| CER | Headline c64 0.0036 – 0.0038; single-run table c64 0.0040; c1 0.0248 |
| Versus initial baseline | **3.29× faster** (A-B-B-A: 2042 → 621 ms) |

### 1.2 Quality priority — profile B3

| Item | Value |
|---|---|
| Content | B2b + DFlash speculative decoding (draft 8) |
| Throughput (concurrency 64) | Headline about **65 req/s**; single-run table **81.56 req/s** |
| Latency (concurrency 64) | **958 – 1628 ms** (A-B-B-A mean 1293 ms); single-run table 769 ms |
| Single-concurrency latency | **62 ms** |
| CER | **Reported as a constant 0.0036 at all levels** |
| Versus initial baseline | 1.40× (A-B-B-A cross-validation) |

### 1.3 How to choose

| Scenario | Choice | Reason |
|---|---|---|
| High-concurrency batch transcription | **B2b** | Substantially higher throughput (single-run table c32 is 2.4× B3); c64 CER differs by only 0.0004 |
| Low concurrency / accuracy-sensitive | **B3** | c1 latency 1.84× faster (114 → 62 ms); c1 CER 6.9× better (0.0248 → 0.0036) |
| External latency commitments | **Both require caution** | B3 segment spread 51.8%, B2b 30.6%; see §9 |

> **Headline and single-run scopes differ**: the headline throughput ranges (B2b 87–119, B3 about 65 req/s) and the single-run table (122.10, 81.56 req/s) come from different measurement scopes. They are retained separately and not combined into one point estimate. The same applies to CER (headline c64 0.0036–0.0038 vs single-run table 0.0040).

---

## 2. Test environment

### 2.1 Hardware

| Item | Test environment |
|---|---|
| Accelerator | Ascend A3 (Atlas 800I A3) |
| NPU allocation | **One die** (all reported measurements were taken under this condition) |
| Host CPU | **Exclusive** during testing; avoid competing host workloads (the server uses many CPU threads for scheduling) |
| Other host specs | Core count / memory not separately recorded for this experiment |

### 2.2 Software

| Component | Version |
|---|---|
| OS | openEuler 22.03 LTS-SP4 (aarch64) |
| Kernel | 5.10.0-216.0.0.115 or later |
| NPU driver | 26.0.rc1 (ascendhal 7.35.23) |
| CANN | 9.1.0 |
| Python | 3.12.x |
| torch / torch_npu | 2.10.0 / 2.10.0.post6 |
| **sglang** | **0.5.23** |
| **sglang-omni** | **0.2.0** |
| transformers | 5.19.0 |
| soundfile / librosa | 0.13.1 / 1.0.1 |
| **jiwer / zhon** | **4.0.0 / 2.0.2** (required for benchmark accuracy evaluation) |

> All data was measured under `sglang 0.5.23 + sglang-omni 0.2.0 + CANN 9.1.0 + driver 26.0.rc1`; other version combinations are not guaranteed to match.
>
> **jiwer and zhon are not in the default image**; without them the benchmark exits immediately with `ModuleNotFoundError: No module named 'jiwer'` (the install command is included in deployment, §6).

---

## 3. Model preparation

| Role | Model | Notes |
|---|---|---|
| Target | `Qwen/Qwen3-ASR-1.7B` | Transcription model; required by B0/B2b/B3 |
| Draft | `Qwen3-ASR-1.7B-dflash` | **Required by B3 only**, for DFlash speculative decoding |

```bash
export MODEL_DIR=/models/Qwen3-ASR-1.7B
export DRAFT_DIR=/models/Qwen3-ASR-1.7B-dflash

# Directories should contain config.json / tokenizer* / *.safetensors, etc.
ls "$MODEL_DIR" "$DRAFT_DIR"
```

> The draft must be compatible with the target and the DFlash implementation, otherwise the speculative acceptance rate drops significantly. Keep the effective context within the draft's **32768-token** attention window. The report supplies local model directories without a pinned draft repository/revision; do not treat an earlier PR's draft version as the confirmed identity of this experiment. See the [Qwen3-ASR cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/qwen3_asr.md#ascend-npu) for mainline support and examples.

---

## 4. Configuration files

All three configurations are complete, copy-paste-ready YAML. B2b / B3 add changes on top of B0 step by step (the stacking logic and attribution are in §5). Replace `<MODEL_DIR>` / `<DRAFT_DIR>` with actual paths.

### 4.1 Initial baseline B0 (tuning starting point)

Save as `b0.yaml`. This is the starting point for all stacking experiments and a common default configuration.

```yaml
config_cls: Qwen3ASRPipelineConfig
name: baseline
model_path: <MODEL_DIR>
stages:
  asr:
    factory:
      dtype: bfloat16
      enable_encoder_cuda_graph: false
      enable_pre_lm_encoder: false        # dedicated encoder service not enabled
    engine:
      max_running_requests: 64
      disable_cuda_graph: false
      cuda_graph_backend_decode: full
      cuda_graph_backend_prefill: disabled  # prefill graph not enabled
      cuda_graph_max_bs: 64
      cuda_graph_bs: [1, 2, 4, 8, 16, 24, 32, 64]
      enable_torch_compile: false
      attention_backend: ascend
      mem_fraction_static: 0.9
      context_length: 32768
      page_size: 128
      chunked_prefill_size: 4096
      mm_attention_backend: ascend_attn
      # no speculative_* section (i.e. no DFlash)
    tp_size: 1
```

> Baseline measurements: c1 128 ms / c64 2094 ms, c64 throughput 30.20 req/s, CER 0.0036–0.0226.

### 4.2 Profile B2b (throughput priority)

Save as `b2b.yaml`. Relative to B0, one change in `factory` and one in `engine` (marked ★).

```yaml
config_cls: Qwen3ASRPipelineConfig
name: b2b
model_path: <MODEL_DIR>
stages:
  asr:
    factory:
      dtype: bfloat16
      enable_encoder_cuda_graph: false
      enable_async_decode: false          # ★ key: 1.85× on its own
      enable_pre_lm_encoder: true         # ★ key: 2.64× on its own
      pre_lm_max_batch_size: 16
      pre_lm_max_batch_wait_ms: 2
      pre_lm_cache_max_entries: 1024
      pre_lm_cache_size_bytes: 2147483648
    engine:
      max_running_requests: 64
      disable_cuda_graph: false
      cuda_graph_backend_decode: full
      cuda_graph_backend_prefill: breakable   # ★ only 1.02×, optional
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

### 4.3 Profile B3 (quality priority)

Save as `b3.yaml`. Adds the speculative-decoding section on top of B2b (marked ★); requires the draft model.

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
      speculative_num_draft_tokens: 8         # ★ do not set 16; measured slower
    tp_size: 1
```

### 4.4 Parameter placement (wrong level fails startup)

| Parameter | Correct level | Error if misplaced |
|---|---|---|
| `enable_async_decode` | `factory:` | `TypeError: Unexpected keyword argument` |
| `enable_pre_lm_encoder` / `pre_lm_*` | `factory:` | same as above |
| `request_build_*` / `prefill_coalesce_*` | `factory:` | same as above |
| `speculative_*` / `cuda_graph_*` / `page_size` / `mem_fraction_static` | `engine:` | — |

> Reason: fields under `engine:` are passed straight through to the underlying `ServerArgs(**kwargs)`, while `enable_async_decode` and similar are factory-level parameters that do not exist in `ServerArgs`.

---

## 5. Tuning method (the most important section)

### 5.1 Four rules that must be followed

| # | Rule | Consequence of violation (measured) |
|---|---|---|
| 1 | **Start from the initial baseline, stack one change at a time** | Starting from a configuration that already includes DFlash counts DFlash's existing effect into later items, breaking attribution |
| 2 | **Changing two items at once masks each other** | Changing "add DFlash" and "disable async decode" together points in opposite directions and nearly cancels, misread as "only 1.01×" |
| 3 | **A-B-B-A cross-validation or ≥2 repeated segments is required** | A single-segment speedup can be overestimated by **1.9×** (measured 2.72× → 1.40×) |
| 4 | **≥320 samples per level, with a fixed request count per level** | Too few samples make the curve non-monotonic and the conclusion unreliable |

### 5.2 Ladder design

| Code | Change | Purpose |
|---|---|---|
| **B0** | Initial baseline | Reference point |
| **B1** | + dedicated encoder service (`enable_pre_lm_encoder: true` + `pre_lm_*`) | Isolate the encoder service |
| **B2** | + prefill CUDA graph (`breakable`) | Isolate the prefill graph |
| **B2b** | + `enable_async_decode: false` | Isolate async decode |
| **B3** | + DFlash (draft 8) | Isolate speculative decoding |

> Note B2b: this step **does not add DFlash**; it specifically separates "disable async decode" from "add DFlash". Without this step the conclusion is wrong.

### 5.3 How to run A-B-B-A

```
Segment 1: B0        ← A
Segment 2: candidate ← B
Segment 3: candidate ← B
Segment 4: B0        ← A
```

Each segment must: **restart the server → 16 serial warmup requests → benchmark → record**. Then compare:

```
candidate mean = (segment 2 + segment 3) / 2
baseline mean  = (segment 1 + segment 4) / 2
speedup        = baseline mean / candidate mean
segment spread = |segment 2 - segment 3| / candidate mean × 100%
```

**The segment spread is itself part of the conclusion**: a spread >30% means the profile is unsuitable as a committed value.

### 5.4 Tuning log (per-item attribution)

| Step | c64 latency | Factor | Note |
|---|---:|---:|---|
| B0 initial baseline | 2094 ms | — | Reference |
| B1 + encoder service | 792 ms | **2.64×** | Largest single-item gain |
| B2 + prefill graph | 774 ms | 1.02× | Very small gain |
| B2b + async decode off | 508 ms | **1.85×** | Second big lever |
| B3 + DFlash | ~1293 ms | 0.39× (slower) | Trades for accuracy, see §8.3 |

> The table above is single-segment observation per stacked item, used for attribution; externally reported speedups use the A-B-B-A cross-validated values in §9.

---

## 6. Deployment

### 6.1 Start the container and server

Prepare an image with the reported runtime and Omni dependencies, then replace `<sglang-omni-image>` (the report does not give an exact pullable image identity). Save the YAML to `/models/` on the host and put audio/manifest under `/data`. `--shm-size=500g` is the reported setting (not proven to be a minimum), but **the default 64 MB causes startup failure** and must be enlarged. Adjust device mappings and driver paths to the host.

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

# Benchmark accuracy dependencies (missing from the default image)
docker exec sglang-omni-asr bash -lc 'python -m pip install jiwer zhon'

# Start the server (choose b0/b2b/b3.yaml for the target profile)
docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/b2b.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'
```

> `ASCEND_RT_VISIBLE_DEVICES` selects the visible die. Privileged mode exposes devices; use it only on a trusted host. See the installation guide for restricted device-access deployment.
>
> **The model name must be consistent**: the server `--model-name`, individual requests and the benchmark must use the same model name (this document uses `Qwen/Qwen3-ASR-1.7B` throughout).

### 6.2 Readiness check and single transcription

```bash
# Readiness (first startup about 2–4 minutes, affected by loading and graph capture)
curl -fsS http://127.0.0.1:8899/v1/models

# Single transcription to confirm inference works
curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
  -F file=@/data/audio/sample.wav \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=zh
```

### 6.3 Interface

| Item | Value |
|---|---|
| Endpoint | `POST /v1/audio/transcriptions` |
| Request format | `multipart/form-data`, audio in the `file` field |
| Relevant fields | `model`, `language`, `stream` |
| Streaming response | SSE events `transcript.text.delta` / `transcript.text.done` |

Example streaming response:

```
data: {"type": "transcript.text.delta", "delta": "营"}
data: {"type": "transcript.text.delta", "delta": "造"}
data: {"type": "transcript.text.done", "text": "营造宁静和平的网络安全空间。"}
data: [DONE]
```

---

## 7. Benchmark procedure

### 7.1 Dataset and scope

The manifest is pipe-delimited per line: `<sample_id>|<reference_text>|<absolute_audio_path>`

```
000000|营造宁静和平的网络安全空间。|/data/audio/000000.wav
000001|今天天气很好。|/data/audio/000001.wav
```

> This document uses the public **Seed-TTS-Eval `zh`** dataset (24 kHz mono, mean 4.79 s).
> **The same reference audio is paired with multiple target texts, so the line count far exceeds the unique-audio count** (640 lines but only 320 unique audios).
> Deduplication significantly changes results; **reports must state whether `--unique-audio` was used**.
> **This document uses the default with-duplicates scope (no `--unique-audio`)**, reporting 640 requests per level. This differs from the deduplicated scope of the 2026-11-20 seven-level record.
> The exact revision, subset selection and manifest hash were not supplied; numerical comparisons require the same manifest.

### 7.2 Benchmark command

Run in a Python environment that can reach the server and `/data` and contains the matching Omni checkout and benchmark dependencies:

```bash
docker exec sglang-omni-asr bash -lc "cd <SGLANG_OMNI_REPO> && python -u \
  -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port 8899 --model-path Qwen/Qwen3-ASR-1.7B \
  --meta /data/meta.lst --lang zh \
  --concurrencies 1,16,32,64 --warmup --stream --repeats 1 \
  --output result.json"
```

| Option | Purpose |
|---|---|
| `--concurrencies` | **Comma-separated string** (space-separated values raise `unrecognized arguments`); levels here are 1,16,32,64 |
| `--warmup` | Built-in warmup; recommended |
| `--stream` | Streaming requests, collecting TTFT and ITL (text-response metrics, not NPU kernel timings) |
| `--unique-audio` | Deduplicates and changes scope; **not used here**; if used, it must be stated |
| `--repeats` | The single-run table uses 1; **for stability use ≥2 and take the median, or run A-B-B-A** (these profiles have high instability risk) |
| `--max-samples` | **Not recommended below 320**, otherwise the curve is non-monotonic |

### 7.3 Additional warmup (required)

In addition to `--warmup`, send 16 serial requests before formal measurement to bring the server to steady state:

```bash
head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done
```

> Without warmup the first level is inflated several-fold (graph-capture overhead is counted into the first request). Retain server logs to confirm graph capture and the effective execution mode.

### 7.4 Conditions during testing

| Requirement | Reason |
|---|---|
| **Exclusive CPU** | The server uses many CPU threads for scheduling; CPU contention significantly raises latency |
| **One instance per die** | Run only one server on the selected die |
| No other workloads | Avoid other processes using NPU memory or PCIe bandwidth |
| Record NPU temperature/frequency | Used to diagnose cross-session drift |
| Repeated segments | See A-B-B-A in §5.3 |

---

## 8. Measured data

> Conditions: Ascend A3 **one die**, exclusive host CPU, fully warmed up, Seed-TTS-Eval `zh` default with-duplicates scope, 640 requests per level. The per-level tables below are **single-run results**; headline ranges and cross-validated values are in §1 and §9.

### 8.1 Three-profile latency and accuracy (single run)

| Concurrency | B0 baseline [latency / CER] | **B2b (throughput)** [latency / CER] | **B3 (quality)** [latency / CER] |
|---:|---|---|---|
| 1 | 128 ms / 0.0226 | **114 ms / 0.0248** | 62 ms / 0.0036 |
| 16 | 662 ms / 0.0067 | **332 ms / 0.0119** | 450 ms / 0.0036 |
| 32 | 1404 ms / 0.0075 | **494 ms / 0.0048** | 1139 ms / 0.0036 |
| 64 | 2094 ms / 0.0036 | **508 ms / 0.0040** | 769 ms / 0.0036 |

> B3 c64 single run is 769 ms; the two A-B-B-A segments are 958–1628 ms (mean 1293 ms), see §9.

### 8.2 Throughput (req/s, single run)

| Concurrency | B0 | **B2b** | **B3** |
|---:|---:|---:|---:|
| 1 | 7.80 | **8.79** | 15.96 |
| 16 | 23.96 | **47.80** | 35.37 |
| 32 | 22.62 | **63.67** | 26.43 |
| 64 | 30.20 | **122.10** | 81.56 |

> B3 throughput is a single-segment value (its latency is unstable, so throughput fluctuates accordingly). The headline B2b 87–119 and B3 about 65 req/s differ in scope from this table and are retained separately.

### 8.3 The DFlash trade-off (important)

| Concurrency | No DFlash (B2b) | With DFlash (B3) | Conclusion |
|---:|---|---|---|
| 1 | 114 ms / CER 0.0248 | **62 ms / CER 0.0036** | DFlash wins outright |
| 16 | **332 ms / CER 0.0119** | 450 ms / CER 0.0036 | DFlash loses on latency, wins on accuracy |
| 64 | **508 ms / CER 0.0040** | 1293 ms / CER 0.0036 | Accuracy differs by only 0.0004, latency by about 2.5× |

**Mechanism (why DFlash does not speed up high concurrency)**: speculative decoding trades compute for bandwidth. Verifying 8 draft tokens at once computes **9 positions**; with an acceptance length of 5.498, **compute per token is about 1.64× the original** (9 / 5.498).

| Scenario | Bottleneck | DFlash effect |
|---|---|---|
| Low concurrency (c1) | Memory bandwidth | **Effective** (1.84×) |
| High concurrency (c16+) | Compute saturation | **Ineffective or even slower** |

> The acceptance length 5.498 comes from the historical eager/graph A-B-B-A report cited in the roadmap (about 4.498 accepted proposals plus one bonus, reported acceptance rate 64.2%); this three-profile experiment did not separately observe the acceptance rate.

### 8.4 CER attribution (isolated and verified)

| Profile | Async decode | DFlash | c1 CER | c16 CER | c32 CER | c64 CER |
|---|---|---|---:|---:|---:|---:|
| B2 | on | no | 0.0248 | 0.0155 | 0.0040 | 0.0044 |
| **B2b** | **off** | no | **0.0248** | **0.0119** | 0.0048 | 0.0040 |
| **B3** | off | **yes** | — | — | **0.0036** | **0.0036** |

**B2 and B2b have identical c1 CER (0.0248) → disabling async decode does not affect accuracy.** Only after adding DFlash does it drop to 0.0036, so **the CER improvement genuinely comes from DFlash**.

---

## 9. Stability and cross-validation (must be reported)

A-B-B-A cross-measurement results (two segments per profile):

| Profile | c32 two segments (ms) | c64 two segments (ms) | c64 segment spread |
|---|---|---|---:|
| **B0 initial baseline** | 1205 / 1058 | 1984 / 2100 | **5.7% (most stable)** |
| **B2b (throughput)** | 545 / 499 | 716 / 526 | **30.6%** |
| **B3 (quality)** | 956 / 1222 | 958 / 1628 | **51.8% (least stable)** |

Corresponding speedups (**cross-validated values, not single-segment**):

| Profile | c32 (×) | c64 (×) |
|---|---:|---:|
| B2b vs B0 | 2.17× | **3.29×** |
| B3 vs B0 | 1.07× | 1.40× |

**Single-segment vs cross-validated difference (why cross-validation is required)**:

| Profile | Single-segment speedup | Cross-validated value | Overestimation |
|---|---:|---:|---:|
| B2b | 4.12× | **3.29×** | 1.25× |
| B3 | 2.72× | **1.40×** | **1.94×** |

> Profiles with segment spread >30% (B2b, B3) are unsuitable as external committed values; B0 is only 5.7%. When choosing a profile, consider throughput, accuracy, tail latency and reproducibility together.

---

## 10. Ineffective and harmful configurations (avoid repeating these mistakes)

| Attempt | Result | Note |
|---|---|---|
| `page_size: 64` | ❌ **CER collapses to 8+** | Output almost entirely wrong; **never usable** |
| `speculative_num_draft_tokens: 16` | ❌ c64 goes 769 → 1137 ms (0.68×) | Keep 8 |
| `max_running_requests: 80` | ❌ Startup failure | `cuda_graph_max_bs must cover max_running_requests (64 < 80)` |
| `prefill_coalesce_*` tuning | ⚠️ No stable gain | Inconsistent direction; noise |
| `request_build_max_workers` 8/16/32 | ⚠️ No improvement | Default 1; larger is worse |
| `mem_fraction_static: 0.92` | ⚠️ Within noise | Not recommended |
| `enable_torch_compile: true` | ❌ Decode-graph capture failure | Must be false |

### 10.1 Two additional constraints

**① `max_running_requests` is constrained by CUDA-graph bucket coverage**: to raise concurrency you must also raise `cuda_graph_max_bs` and extend the `cuda_graph_bs` buckets, otherwise startup fails immediately (see table above).

**② `prefill_coalesce_wait_ms` must be > 0**:

```yaml
prefill_coalesce_requests: 0        # disable coalescing
prefill_coalesce_wait_ms: 0.1       # ✅ cannot be 0.0 (schema requires gt=0)
```

---

## 11. Reproduction checklist

| # | Check | Consequence if missing |
|---|---|---|
| 1 | Target model present; (B3) draft model also present | DFlash cannot start |
| 2 | **`enable_async_decode: false` under `factory:`** | Loses 1.85× throughput; error if under engine |
| 3 | **`enable_pre_lm_encoder: true` + `pre_lm_*` under `factory:`** | Loses 2.64× throughput |
| 4 | `speculative_num_draft_tokens: 8` | Setting 16 is 1.48× slower |
| 5 | **`page_size` stays 128** | Setting 64 collapses CER to 8+ |
| 6 | `enable_torch_compile: false` | Decode-graph capture failure |
| 7 | `cuda_graph_bs` covers the target maximum concurrency | High concurrency degrades to eager |
| 8 | `--shm-size` enlarged | Startup failure |
| 9 | `jiwer` + `zhon` installed | Benchmark exits immediately |
| 10 | `--concurrencies` comma-separated | Argument parsing failure |
| 11 | Warmup completed (16 serial + `--warmup`) | First level inflated several-fold |
| 12 | No other host load during testing, one die exclusive | Latency significantly raised |
| 13 | **A-B-B-A or repeated segments done** | Single segment can overestimate by 1.9× |
| 14 | Whether `--unique-audio` was used is recorded (not used here by default) | Scopes not comparable |
| 15 | ≥320 samples per level | Non-monotonic curve |

---

## 12. Known limits

| Item | Note |
|---|---|
| **Beyond upstream qualified scope** | Upstream qualified only decode buckets [1,2] with encoder/prefill graphs disabled; this document enables the prefill graph and extends buckets to 64, so **correctness must be revalidated independently** |
| Candidate profiles drift | B2b segment spread 30.6%, B3 51.8%; initial baseline only 5.7% |
| One die | Multi-die scaling untested; cannot linearly extrapolate to the whole machine |
| No acceptance-rate observation | This run cannot obtain the acceptance rate, so draft count cannot be tuned further (the 5.498 in §8.3 is cited from a historical report) |
| Sample size | 640 requests per level; cross-validation only 2 segments per profile |
| Audio length | Mean 4.79 s; >30 s not covered |
| Language | Chinese only |
| Concurrency cap | Up to 64 |
| Peak device memory | Not recorded |
| Reproduction identity | Exact image, code/model revisions, manifest and raw per-request results remain to be recorded |

---

## 13. Minimal command summary

These commands assume the image, models, YAML, manifest and audio have been prepared as above. The benchmark environment must be able to reach the server and `/data`.

```bash
# 1. Start (choose b0/b2b/b3.yaml for the target profile)
docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/b2b.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'

# 2. Wait for readiness
until curl -fsS -o /dev/null http://127.0.0.1:8899/v1/models; do sleep 5; done

# 3. Warmup (16 serial requests)
head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done

# 4. Benchmark (single-run scope: with duplicate audio, no --unique-audio)
docker exec sglang-omni-asr bash -lc "cd <SGLANG_OMNI_REPO> && python -u \
  -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port 8899 --model-path Qwen/Qwen3-ASR-1.7B \
  --meta /data/meta.lst --lang zh \
  --concurrencies 1,16,32,64 --warmup --stream --repeats 1 --output result.json"
```

> **Reproducing speedups and spread**: run A-B-B-A per §5.3 (B0 → candidate → candidate → B0), restarting the server and re-warming up each segment, then compute the cross-validated speedup and segment spread with the formulas. A single segment overestimates the speedup (B3 by up to 1.94×).

---

## 14. One-page quick reference

```yaml
# Throughput priority (B2b): minimal change set relative to B0
factory:
  enable_async_decode: false          # +1.85×
  enable_pre_lm_encoder: true         # +2.64×
  pre_lm_max_batch_size: 16
  pre_lm_max_batch_wait_ms: 2
  pre_lm_cache_max_entries: 1024
  pre_lm_cache_size_bytes: 2147483648
engine:
  cuda_graph_backend_prefill: breakable   # +1.02×

# Quality priority (B3) = B2b plus:
engine:
  speculative_algorithm: DFLASH            # CER → 0.0036
  speculative_draft_attention_backend: ascend
  speculative_draft_model_path: <DRAFT_DIR>
  speculative_num_draft_tokens: 8
```

**Do not change**: `page_size` (keep 128), `enable_torch_compile` (keep false), `speculative_num_draft_tokens` (keep 8).