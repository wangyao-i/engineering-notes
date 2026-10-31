# Qwen3-ASR-1.7B on Ascend A3: Performance Results, Graph Optimization, and Reproduction

[中文版](qwen3-asr-ascend-performance.zh-CN.md) | Updated: 2026-10-31

We tested offline Qwen3-ASR-1.7B serving on Atlas A3 (910C), focusing on throughput, useful configuration changes, and recognition quality.

There are two distinct sets of results. A reference recipe with encoder graphs and eager LM execution achieved 1.2031% English WER and 0.5933% Chinese CER on full SeedTTS-Eval. Separately, FIA decode-graph experiments achieved approximately 2.54–2.67× the throughput of the same FIA path with decode graphs disabled in paired fixed-input tests. The graph configuration also showed higher corpus-level error rates in an independent quality comparison. We report both the performance gain and the quality difference; the graph configuration is not yet a correctness-qualified default recommendation.

## 1. Environment and measurement definitions

| Item | Configuration |
|---|---|
| Model | Qwen3-ASR-1.7B, revision `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| Hardware | Atlas A3 / 910C; 2 dies per physical card, 64 GB HBM per die |
| SGLang-Omni | 0.1.9, `72b776c9ce41c794e74fa609266915ad229bea3c` |
| SGLang | 0.5.22 |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0.post6 |
| CANN / driver / firmware | 9.1.0 / 25.5.0 / 7.8.0.5.216 |
| dtype | Configured as `auto`; the resolved dtype was not separately exported |

| Experiment group | Execution configuration | Input and purpose |
|---|---|---|
| Reference recipe | Encoder graphs on, LM eager, async decode on, engine compile off, FIA=False, pre-LM encoder on | SeedTTS-Eval performance, quality, and bounded sustained runs |
| Optimization experiments | FIA=1; decode graphs on/off; pre-LM encoder off; memory fraction 0.25 | Fixed 10-second loop-concatenated audio; combined tuning and individual feature comparisons |
| Multi-instance example | 8 dies, 16 instances, decode graphs on, client-side least-outstanding dispatch | Short closed-loop concurrency-70 run with response-validity checks |

The reference recipe follows the A-only profile from [PR #3199](https://github.com/sgl-project/sglang-omni/pull/3199), reproduced on a different dependency stack. Its performance and sustained-run groups use `max_running_requests=8`, while the quality group uses 64. These are not joint acceptance results for one configuration. Although the YAML requested radix-cache disabling, initialization still appeared in logs; the effective disabled state remains unconfirmed.

Request latency starts immediately before HTTP sending and ends after reading the complete response body, including upload and queueing. Completed-request throughput divides the formal request count by the interval from first send to last completion. Closed-loop concurrency replenishes completed requests; it is neither a single simultaneous burst nor a count of realtime sessions.

## 2. Baseline versus combined tuning

On the same single-die, two-instance deployment, the runnable baseline used ATB, decode graphs off, pre-LM encoding enabled by default, and default coalescing. The tuned configuration used FIA and decode graphs, disabled pre-LM encoding, and set coalescing to 10 ms / 32 requests. Both used the same fixed 10-second input. Each concurrency point has three rounds; entries are medians of the per-round metrics.

| Concurrency | Baseline req/s | Tuned req/s | Baseline p95 | Tuned p95 |
|---:|---:|---:|---:|---:|
| 1 | 0.90 | 3.75 | 1230 ms | 305 ms |
| 8 | 6.68 | 17.5 | 1301 ms | 472 ms |
| 32 | 13.55 | 28.5 | 2574 ms | 1167 ms |

Across three rounds, baseline throughput ranges were 0.86–0.95, 6.62–6.76, and 13.41–13.71 req/s; tuned ranges were 3.48–3.79, 17.1–17.8, and 28.4–28.6 req/s. Combined tuning improved throughput and p95 at all three points, but latency still rose substantially at concurrency 32. Multiple settings changed together, so this is not an isolated feature attribution. HTTP success does not establish equivalent recognition quality.

## 3. Which optimizations mattered?

### Decode graphs: the largest measured performance change

On 8 dies / 16 instances, we held FIA, pre-LM encoding, memory fraction, request limit, load balancing, and warmup constant while switching the decode graph backend. Each arm ran twice at closed-loop concurrency 70, with 700 requests per round.

| Paired round | Throughput: non-graph → graph | Ratio | p50: non-graph → graph | p95: non-graph → graph |
|---|---|---:|---|---|
| 1 | 48.84 → 130.47 req/s | 2.67× | 1259.1 → 490.6 ms | 1644.4 → 568.4 ms |
| 2 | 51.03 → 129.80 req/s | 2.54× | 1270.2 → 510.9 ms | 1545.5 → 600.0 ms |

Both rounds moved in the same direction, with approximately 60% lower p50. Capture logs and the corresponding replay code path were checked, but there are no runtime per-request replay counts. We therefore describe this as a paired comparison of decode-graph configurations. It covers short fixed-input runs; the separate quality comparison follows below.

### Coalescing: a tradeoff between waiting and batching

Reducing the coalescing wait from 40 ms to 10 ms in the same graph-mode deployment changed throughput from 117.8/125.6 to 130.5/129.8 req/s in two paired comparisons, approximately 3.3%–10.7% higher. One round improved p95 and the other worsened it. A 10 ms wait was useful for this workload, but shorter waits are not universally better.

### Instances and dispatch: tail latency and load distribution

| Comparison | Observation | Interpretation |
|---|---|---|
| 1 → 2 instances per die | p95 814/846 → 568/600 ms; throughput 124.4/128.0 versus 130.5/129.8 req/s | Two instances improved tail latency here; one instance had lower p50 |
| Round-robin → least-outstanding | One comparison: 98.5 → 129.8 req/s; p95 865 → 600 ms | Dispatch matters; internal queueing causality was not established |
| Encoder graphs on/off | Throughput differences of approximately 2.8%–3.3%; inconsistent p50 direction | Insufficient evidence for a stable benefit |
| CPU affinity | Throughput differences of approximately 0.7%–0.9% | An isolation option; actual NPU-to-NUMA affinity was not verified |

Disabling the pre-LM encoder changes where encoding runs, batching, and caching behavior. This round did not isolate its benefit. It does not disable the encoder itself or necessarily disable encoder graphs.

## 4. Does faster execution preserve recognition quality?

The optimization experiments also compared LM eager and decode-graph configurations on full SeedTTS-Eval, using the same normalization and word-level English / character-level Chinese scoring.

| Paired quality configurations | English WER, 1088 records | Chinese CER, 2020 records |
|---|---:|---:|
| Earlier LM eager configuration | 1.2107% (160/13216) | 0.6113% (204/33370) |
| Tuned decode-graph configuration | 1.4755% (195/13216) | 0.7042% (235/33370) |

The graph configuration added 35 English and 31 Chinese edit errors in this comparison. Both arms had zero HTTP failures and empty outputs, yet corpus-level error rates increased. Sample-level paired summaries in the source materials contain unresolved numerical inconsistencies, so we do not claim sample-level superiority or quality equivalence. These are configuration-level differences; attributing individual errors requires further reproduction and path investigation.

A separate reference-recipe quality run following PR #3199 used C4 and `max_running_requests=64`: **1.2031% EN WER** (159/13216) and **0.5933% ZH CER** (198/33370), with zero failures or empty outputs. It does not qualify the graph-mode configuration above. The dataset includes duplicate audio records; all records are retained in quality denominators to follow the public evaluation convention.

## 5. Public-dataset performance and sustained runs

The reference-recipe performance matrix used 100 fixed SeedTTS clips per language spanning short, medium, and long duration bands, averaging approximately 5.07 seconds. A single die and one service process were used. C1/C4 each ran three rounds with a fresh process and excluded warmup: all 1200 formal requests succeeded without empty text.

| Language / concurrency | req/s | audio-s/s | p50 | p95 |
|---|---:|---:|---:|---:|
| EN C1 | 1.931 | 9.79 | 508.7 ms | 690.8 ms |
| EN C4 | 5.996 | 30.39 | 655.1 ms | 869.7 ms |
| ZH C1 | 2.104 | 10.67 | 456.4 ms | 633.1 ms |
| ZH C4 | 6.279 | 31.85 | 614.7 ms | 828.9 ms |

These are medians of per-round metrics; p95 is not recomputed from 300 pooled samples. This group uses `max_running_requests=8`, pre-LM encoding on, and automatic memory budgeting, unlike the graph-optimization group's 0.25 memory fraction.

C4 sustained runs used real short/medium/long audio plus separately marked 17–24-second concatenations, stopping at a request limit:

| Language | Actual duration | Completed requests | Throughput | Failures / empty outputs |
|---|---|---:|---:|---|
| EN | 31.04 minutes | 10003 | 5.37 req/s | 0/0 |
| ZH | 29.79 minutes | 10003 | 5.60 req/s | 0/0 |

Inputs cycled and cache hits were observed, so this was not a sustained unique-input workload. Process memory rose from approximately 52 GB to 54–55 GB; the cause is unconfirmed. At C8, the same reference recipe hung twice with no completed requests, once with `max_running_requests=64` and once with 8, while health checks still returned healthy. Coverage is therefore limited to C1/C4 and approximately 30-minute runs; reducing the request limit is not a verified fix.

## 6. Multi-instance example and reproduction entry point

The B-v2 run used fixed 10-second audio, 8 dies / 16 instances, and closed-loop concurrency 70. It completed 1400 requests at **135.04 req/s**, with **p50 489.1 ms, p95 623.2 ms, and p99 735.1 ms**. HTTP, JSON parsing, empty-text, and defined degradation-pattern checks found no anomalies in that run. This is an approximately 10-second deployment example, not sustained production capacity or a p95-under-500-ms result.

Choose the recipe by purpose: the reference recipe for public-dataset quality/performance validation, and FIA decode graphs for performance exploration with the quality differences retained. Do not combine their environment settings, cache state, and instance counts into one recommended configuration. The source materials linked below describe recipes and script inventories. Independent downloadable scripts have not yet been published; directory examples in the articles are not a delivered script package.

With the original scripts and input audio available and all 16 instances healthy, the B-v2 command is:

```bash
set -eu
mkdir -p runs
RUN_DIR=$(mktemp -d runs/B-XXXXXXXX)
RUN_CSV="$RUN_DIR/requests.csv" python3 bench/bench_rr.py \
  --ports 8100-8115 --lb pending --audio audio_10s.wav \
  --concurrency 70 --requests 1400 --seconds 10 \
  --warmup-per-port 1 --warmup-log "$RUN_DIR/warmup.log" \
  > "$RUN_DIR/benchmark.log" 2>&1
python3 tools/summarize.py "$RUN_DIR/requests.csv" --json \
  > "$RUN_DIR/summary.json"
```

`--seconds 10` labels audio duration; the run ends after 1400 requests. Warmup is excluded. Check that warmup drained successfully: the original script only warns on timeout, so a run with that warning does not satisfy the warmup condition. Different audio changes duration and output-token distributions and needs a separate result record.

The planned public package prioritizes both serving recipes, closed/open-loop clients, fixed-corpus manifest generation, performance summaries, and quality scoring matching the public evaluation convention. Original measurement scripts and later revisions should retain separate identities.

## 7. Known issues and follow-up

1. **ATB/FIA graph-capture compatibility.** The ATB decode path exhibited an input-contract error and a subsequent `AclmdlRICaptureEnd 107033` failure. FIA could complete capture, but capture success still needs output-correctness validation. Backend/version support and graph-versus-eager comparisons are tracked in [Ascend roadmap #2076: NPU-GRAPH-001](https://github.com/sgl-project/sglang-omni/issues/2076).
2. **Output degradation after realtime overload.** WebSocket overload was followed by degraded offline ASR output on the same instance. [Ascend roadmap #2076: NPU-GRAPH-002](https://github.com/sgl-project/sglang-omni/issues/2076) tracks graph Update/Replay ordering, fixes, and correctness/performance validation. Realtime requires separate testing; the offline results here do not establish realtime capacity or recovery behavior.

Decode-graph configuration produced substantial performance changes, while quality and concurrency boundaries remain unresolved. Coalescing, instance count, and dispatch also affect tail latency. Next steps prioritize publishing runnable scripts and validating graph-mode quality regressions, C8 hangs, and sustained real-speech workloads.

## Sources

This article synthesizes two public server reports in `wangyao-i/engineering-notes`, read on 2026-10-31. The NPU tests were not rerun during article preparation.

- [Community article data package (Chinese)](https://github.com/wangyao-i/engineering-notes/blob/main/sglang-omni/Qwen3-ASR%20on%20Ascend%20A3%20%E2%80%94%E2%80%94%20%E7%A4%BE%E5%8C%BA%E6%96%87%E7%AB%A0%E6%95%B0%E6%8D%AE%E5%8C%85%EF%BC%88Qwen3-ASR-1.7B%EF%BC%8C%E7%AC%AC%E4%B8%80%E7%AF%87%EF%BC%89)
- [Stage performance optimization report (Chinese)](https://github.com/wangyao-i/engineering-notes/blob/main/sglang-omni/Qwen3-ASR-1.7B%20%E5%9C%A8%E6%98%87%E8%85%BE%20A3%20%E4%B8%8A%E7%9A%84%E9%98%B6%E6%AE%B5%E6%80%A7%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5)
