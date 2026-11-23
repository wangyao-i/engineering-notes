# Qwen3-ASR-1.7B on Ascend A3: Tuning Results (2026-11-23)

[中文版](qwen3-asr-ascend-tuning-results-2026-11-23.zh-CN.md) | Updated: 2026-11-23

This note summarizes three Qwen3-ASR-1.7B configurations on one Ascend A3 die and their A-B-B-A latency variation. The figures are operator-reported; no hardware run was repeated for this document.

## Summary

B2b targets throughput at higher concurrency and does not use DFlash. B3 adds DFlash (draft 8) to B2b, favoring low-concurrency latency and recognition quality. Repeated runs vary substantially for both candidates, so these results are not service-level latency guarantees.

| Profile | Main changes | C1 latency | C64 latency | C64 throughput | CER |
|---|---|---:|---:|---:|---|
| B0 baseline | No dedicated encoder service, prefill graph or DFlash | 128 ms | 2094 ms | 30.20 req/s | 0.0036–0.0226 across C1–C64 |
| B2b throughput priority | Dedicated encoder service, breakable prefill graph, async decode off; no DFlash | 114 ms | 526–716 ms | Headline range 87–119 req/s; single-run table 122.10 req/s | C1 0.0248; C64 0.0036–0.0038 |
| B3 quality priority | B2b plus DFlash, 8 draft tokens | 62 ms | 958–1628 ms (A-B-B-A mean 1293 ms) | Headline about 65 req/s; single-run table 81.56 req/s | Reported as 0.0036 at all levels |

The headline throughput ranges and single-run table differ. They are retained as separate reported scopes rather than combined into one point estimate.

## Per-level latency and accuracy

| Concurrency | B0 latency / CER | B2b latency / CER | B3 latency / CER |
|---:|---|---|---|
| 1 | 128 ms / 0.0226 | 114 ms / 0.0248 | 62 ms / 0.0036 |
| 16 | 662 ms / 0.0067 | 332 ms / 0.0119 | 450 ms / 0.0036 |
| 32 | 1404 ms / 0.0075 | 494 ms / 0.0048 | 1139 ms / 0.0036 |
| 64 | 2094 ms / 0.0036 | 508 ms / 0.0040 | 769 ms single run; 958–1628 ms in A-B-B-A |

## Single-run throughput

| Concurrency | B0 (req/s) | B2b (req/s) | B3 (req/s) |
|---:|---:|---:|---:|
| 1 | 7.80 | 8.79 | 15.96 |
| 16 | 23.96 | 47.80 | 35.37 |
| 32 | 22.62 | 63.67 | 26.43 |
| 64 | 30.20 | 122.10 | 81.56 |

B3 throughput varied with latency. The headline estimate of about 65 req/s and the single-run table value of 81.56 req/s retain their separate scopes.

## A-B-B-A repeatability

| Metric | B0 | B2b | B3 |
|---|---:|---:|---:|
| C64 latency observations (ms) | 1984 / 2100 | 716 / 526 | 958 / 1628 |
| C64 run-to-run spread | 5.7% | 30.6% | 51.8% |
| Speedup vs B0 at C32 | — | 2.17× | 1.07× |
| Speedup vs B0 at C64 | — | 3.29× | 1.40× |

B2b reports higher C64 throughput. B3 reports CER 0.0036 at the tested levels and lower C1 latency. Run-to-run variation is substantial for both candidates, particularly B3. Select a profile by considering throughput, accuracy, tail latency and repeatability together.

## Configuration and test scope

| Item | Reported setup |
|---|---|
| Hardware | Ascend A3 (Atlas 800I A3), one die; host CPU kept exclusive during testing |
| Software | openEuler 22.03 LTS-SP4 aarch64; driver 26.0.rc1 / ascendhal 7.35.23; CANN 9.1.0; Python 3.12; torch 2.10.0 / torch_npu 2.10.0.post6; SGLang 0.5.23; SGLang-Omni 0.2.0; transformers 5.19.0 |
| Workload | Chinese Seed-TTS-Eval; 640 requests reported per level; one die; 4.79 s mean audio duration |
| B2b | `enable_async_decode: false`, `enable_pre_lm_encoder: true`, prefill graph `breakable`, no DFlash |
| B3 | B2b configuration with DFlash enabled and 8 draft tokens |
| Cross-run method | A-B-B-A; the two C64 observations per candidate are listed above |

See the [complete test record in the roadmap](https://github.com/sgl-project/sglang-omni/issues/2076) for full configurations, tuning steps, CER attribution, failed configurations and reproduction checks.

## Limits

| Topic | Evidence boundary |
|---|---|
| Repeatability | C64 spread is 30.6% for B2b and 51.8% for B3, versus 5.7% for B0 |
| Graph correctness | The test enables prefill graphs and decode buckets through 64, beyond the original qualification scope of the merged NPU DFlash recipe; revalidate on the target environment |
| Device scaling | One die only; multi-die scaling is untested |
| Samples and workload | 640 requests per level reported, only two A-B-B-A segments per profile; Chinese audio averaging 4.79 s, with long audio out of scope |
| Other | Peak device memory and DFlash acceptance rate were not recorded; pin the exact image, code and model revisions separately |

The earlier seven-level single-run DFlash results remain in the [2026-11-20 bilingual reproduction guide](qwen3-asr-ascend-dflash-reproduction-2026-11-20.en.md); its workload and configuration differ from the three-profile tuning experiment here.
