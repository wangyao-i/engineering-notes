# Qwen3-ASR-1.7B on Ascend A3: DFlash Performance and Reproduction (2026-11-20)

[中文版](qwen3-asr-ascend-dflash-reproduction-2026-11-20.zh-CN.md) | Updated: 2026-11-20

This document is adapted from the [original roadmap report](https://github.com/sgl-project/sglang-omni/issues/2076#issuecomment-7032927746). It records the DFlash decode-graph configuration, deployment procedure and seven concurrency levels for Qwen3-ASR-1.7B on one Ascend A3 die. Results were reported by the test operator; preparing this document did not involve another hardware run.

The experiment used SGLang 0.5.23, SGLang-Omni 0.2.0, CANN 9.1.0 and driver 26.0.rc1. The original report does not identify the image tag/digest, source commit, target/draft revisions, exact 640-clip manifest or raw per-request results. This is a reproduction procedure for the reported configuration, rather than a guarantee of identical measurements from package versions alone. See the [NPU installation guide](https://github.com/sgl-project/sglang-omni/blob/main/docs/other_platforms/installation_npu.md) for environment setup; its current mainline recommendations do not identify this experiment's stack.

## 1. Results at a glance

| Item | Reported result |
|---|---|
| Configuration | DFlash speculative decoding, maximum running requests 64 |
| Device use | One die |
| Highest throughput in this sweep | 76.26 req/s at concurrency 64 |
| Concurrency 64 latency | Mean 825 ms, P50 504 ms, P95 3622 ms |
| Recognition quality | Chinese CER 0.0036 (0.36%) at all seven levels |
| Aggregate RTFx | 60.1 at concurrency 1; 364.9 at concurrency 64 |

## 2. Reported test environment

### Hardware

| Item | Test environment |
|---|---|
| Accelerator | Ascend A3, Atlas 800I A3 |
| NPU allocation | One die for all reported measurements |
| Host CPU | 640 logical cores; avoid competing host workloads during testing |
| Host memory | 2 TB on the test machine; this is not an established minimum requirement |

### Software

| Component | Reported version |
|---|---|
| OS | openEuler 22.03 LTS-SP4, aarch64 |
| Kernel | 5.10.0-216.0.0.115 |
| NPU driver | 26.0.rc1, ascendhal 7.35.23 |
| CANN | 9.1.0 |
| Python | 3.12.x |
| torch / torch_npu | 2.10.0 / 2.10.0.post6 |
| SGLang / SGLang-Omni | 0.5.23 / 0.2.0 |
| transformers | 5.19.0 |
| soundfile / librosa | 0.13.1 / 1.0.1 |
| jiwer / zhon | 4.0.0 / 2.0.2, used by benchmark accuracy evaluation |

Other version combinations have not been qualified by this report.

## 3. Model preparation

| Role | Model | Purpose |
|---|---|---|
| Target | `Qwen/Qwen3-ASR-1.7B` | Transcription model |
| Draft | `Qwen3-ASR-1.7B-dflash` | Required for DFlash speculative decoding |

```bash
export MODEL_DIR=/models/Qwen3-ASR-1.7B
export DRAFT_DIR=/models/Qwen3-ASR-1.7B-dflash

ls "$MODEL_DIR" "$DRAFT_DIR"
```

Prepare the model files, including their configuration and safetensors. The draft must be compatible with the target and the DFlash implementation. The report supplies local paths without an exact draft repository/revision; an earlier PR's pinned draft must not be treated as the confirmed identity of this experiment. Consult the [Qwen3-ASR cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/qwen3_asr.md#ascend-npu) for mainline support and examples.

## 4. Configuration used in this test

Save the following as `c64_dflash.yaml`:

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
      max_running_requests: 64
      disable_cuda_graph: false
      cuda_graph_backend_decode: full
      cuda_graph_backend_prefill: disabled
      cuda_graph_max_bs: 64
      cuda_graph_bs: [1, 2, 4, 8, 16, 24, 32, 64]
      enable_torch_compile: false
      attention_backend: ascend
      speculative_algorithm: DFLASH
      speculative_draft_attention_backend: ascend
      speculative_draft_model_path: /models/Qwen3-ASR-1.7B-dflash
      speculative_num_draft_tokens: 8
```

| Setting | Value | Meaning in this experiment |
|---|---|---|
| `speculative_algorithm` | `DFLASH` | Enables DFlash; this sweep has no no-DFlash control and cannot isolate its contribution |
| `speculative_num_draft_tokens` | `8` | Configured value; accepted/bonus lengths were not supplied in this report |
| `enable_torch_compile` | `false` | Retain the tested support configuration; other combinations are outside this report |
| `cuda_graph_backend_decode` | `full` | Enables decode-graph execution |
| `cuda_graph_bs` | `[1,2,4,8,16,24,32,64]` | Eight graph buckets including all measured concurrency levels |
| `max_running_requests` | `64` | Matches the highest tested concurrency |
| `enable_async_decode` | `false` | Synchronous scheduling, consistent with merged NPU DFlash support |

Encoder and prefill graphs are disabled in this profile.

## 5. Deployment

### Start the server

Prepare an image with the reported runtime and Omni dependencies, then replace `<sglang-omni-image>` below. The source report does not give an exact pullable image identity. Save the YAML to `/models/c64_dflash.yaml` on the host and put the manifest/audio under `/data`. The commands below add a `/data` mount and use one consistent model name for serving, individual requests and benchmarking. The 500 GB shared-memory setting is copied from the report and is not an established minimum. Adjust device mappings and driver paths to the host.

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

docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/c64_dflash.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'
```

`ASCEND_RT_VISIBLE_DEVICES` selects the visible die. The original command uses privileged mode to expose devices; use it only on a trusted host. See the installation guide for restricted device-access deployment.

### Check readiness and transcribe one clip

```bash
curl -fsS http://127.0.0.1:8899/v1/models
```

Check that the response includes the configured model, then submit a transcription to confirm inference works. The original report estimates 2–4 minutes for first startup; loading and graph capture affect the actual duration.

```bash
curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
  -F file=@/data/audio/sample.wav \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=zh
```

| Interface item | Value |
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

## 6. Benchmark procedure

### Manifest

Use a pipe-delimited manifest with absolute audio paths:

```
<sample_id>|<reference_text>|<absolute_audio_path>|[optional_fields]
```

```
000000|营造宁静和平的网络安全空间。|/data/audio/000000.wav
000001|今天天气很好。|/data/audio/000001.wav
```

The report uses 640 Chinese Seed-TTS-Eval clips, 24 kHz mono, with a mean duration of 4.79 s. This is not the full Chinese corpus. The exact revision, subset-selection procedure and manifest hash were not supplied; numerical comparisons require the same manifest.

### Benchmark command

Run in a Python environment containing the matching Omni checkout and benchmark dependencies, with access to the server and `/data`:

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

| Option | Purpose |
|---|---|
| `--concurrencies` | A comma-separated string, rather than space-separated values |
| `--unique-audio` | Selects clips using the benchmark's audio-deduplication rules; verify the final count is 640 per level |
| `--warmup` | Performs built-in warmup |
| `--stream` | Sends streaming requests and collects TTFT/ITL; these are text-response metrics, not NPU kernel timings |
| `--lang zh` | Uses Chinese recognition and CER normalization |
| `--repeats 1` | One measured repeat per level |

### Warmup and system load

The source procedure recommends 16 serial requests before the formal run, in addition to `--warmup`:

```bash
head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done
```

Separate loading, capture and warmup from formal measurements. Otherwise first-request costs can enter the first level's statistics. Retain server logs to verify capture and the effective execution mode.

| Condition | Reason |
|---|---|
| Avoid competing CPU work | The server uses CPU threads for scheduling; contention can affect latency |
| One server on the selected die | Avoid device-memory and execution contention |
| Avoid other accelerator/PCIe workloads | Keep resource contention out of the comparison |

## 7. Reported performance

All values below are from the original single-die, warmed-up run with an otherwise idle host CPU. The operator reports 640/640 evaluated clips at each level.

| Concurrency | Mean latency (ms) | P50 (ms) | P95 (ms) | TTFT (ms) | Throughput (req/s) | RTFx | CER (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79 | 77 | 105 | 39.8 | 12.56 | 60.1 | 0.36 |
| 4 | 260 | 125 | 200 | 116.5 | 15.35 | 73.5 | 0.36 |
| 8 | 254 | 142 | 275 | 98.3 | 31.32 | 149.9 | 0.36 |
| 16 | 259 | 177 | 306 | 98.2 | 61.35 | 293.6 | 0.36 |
| 24 | 404 | 275 | 518 | 171.3 | 58.74 | 281.1 | 0.36 |
| 32 | 442 | 323 | 1188 | 224.3 | 71.62 | 342.7 | 0.36 |
| 64 | 825 | 504 | 3622 | 245.0 | 76.26 | 364.9 | 0.36 |

| Concurrency | Mean RTF | Mean ITL (ms) |
|---:|---:|---:|
| 1 | 0.0168 | 14.0 |
| 4 | 0.0547 | 44.4 |
| 8 | 0.0540 | 49.2 |
| 16 | 0.0524 | 49.3 |
| 24 | 0.0831 | 62.7 |
| 32 | 0.0929 | 57.7 |
| 64 | 0.1726 | 123.2 |

Per-request RTF is request duration divided by audio duration; the table reports its mean. Aggregate RTFx is completed audio duration divided by wall-clock benchmark duration. Under concurrency, it is not the reciprocal of mean per-request RTF. CER 0.0036 is displayed as 0.36%.

| Observation | Interpretation |
|---|---|
| Throughput | 12.56 → 76.26 req/s from concurrency 1 → 64, approximately 6.1×. Scaling is not monotonic: concurrency 16 → 24 drops from 61.35 to 58.74 req/s. |
| Mean latency | 79 → 825 ms, approximately 10.4×; consider tail latency when selecting concurrency. |
| Accuracy | Reported CER is unchanged across the seven levels. This sweep has no no-DFlash baseline and does not establish lossless decoding. |
| Tail latency | P95 is at most 306 ms for the measured levels through 16, 1188 ms at 32, and 3622 ms at 64. The report does not establish the cause of the tail. |

Concurrency 64 produced the highest throughput in this sweep, with a larger latency tail. Select a level against the application's latency requirement; this result does not establish a universal optimum.

## 8. Reproduction checklist

| Check | Consequence if missing |
|---|---|
| Target and compatible draft files are present | DFlash cannot initialize |
| `speculative_algorithm: DFLASH` | Configuration differs from this experiment |
| `enable_torch_compile: false` | Outside the tested configuration |
| Graph buckets retained and capture/execution logs checked | The expected graph path is unconfirmed |
| Shared memory sized for the runtime | Insufficient shared memory may affect startup or serving |
| An idle die selected | Memory/execution contention may affect startup or results |
| Reported warmup procedure completed | First-capture costs may enter measured results |
| No competing host work | CPU contention may increase latency |
| Comma-separated concurrency values | Argument parsing can fail |
| Benchmark dependencies installed, including `jiwer` and `zhon` | Evaluation or imports may fail |

## 9. Limits and missing reproduction identity

| Topic | Scope |
|---|---|
| Device scaling | One die only; multi-die scaling was not measured |
| Audio length | Mean 4.79 s; performance on long clips over 30 s is not covered |
| Language | Chinese only in this sweep |
| Concurrency | Tested through 64; higher values are unvalidated |
| Device memory | Peak not recorded; `mem_fraction_static` uses its default |
| Repeatability | One measured repeat per level, without repeated-run spread |
| Identity | Exact image, source/model revisions, manifest and raw request data remain to be recorded |

## 10. Minimal command summary

These commands assume the image, models, YAML, manifest and audio have already been prepared as above. The benchmark environment must be able to reach the service and `/data`.

```bash
docker exec -d sglang-omni-asr bash -lc \
  'sgl-omni serve --config /models/c64_dflash.yaml --model-name Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8899'

until curl -fsS -o /dev/null http://127.0.0.1:8899/v1/models; do sleep 5; done

head -16 /data/meta.lst | awk -F'|' '{print $3}' | while read -r w; do
  curl -s -X POST http://127.0.0.1:8899/v1/audio/transcriptions \
    -F file=@"$w" -F model=Qwen/Qwen3-ASR-1.7B -F language=zh -o /dev/null
done

python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port 8899 --model-path Qwen/Qwen3-ASR-1.7B --meta /data/meta.lst --lang zh \
  --concurrencies 1,4,8,16,24,32,64 \
  --unique-audio --warmup --stream --repeats 1 --output result.json
```
