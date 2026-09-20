# From CANN 107033 to a Silent Stall: Debugging and Redesigning Qwen3-ASR Full-Graph Execution on Ascend

> This article documents an engineering investigation into Qwen3-ASR full-graph execution on Ascend in SGLang-Omni.
>
> - Implementation: [sglang-omni#2160](https://github.com/sgl-project/sglang-omni/pull/2160)
> - Roadmap: [sglang-omni#2076](https://github.com/sgl-project/sglang-omni/issues/2076)
> - 中文版：[qwen3-asr-ascend-graph-debugging.zh-CN.md](qwen3-asr-ascend-graph-debugging.zh-CN.md)
>
> This article explains the investigation and design rationale. The PR description and its Validation section remain authoritative for the current diff and qualification status. The results apply only to the stated versions, configuration, and workload; they are not a universal stability or performance claim.

## Summary

While enabling full-graph execution for Qwen3-ASR on Ascend NPU, we encountered two failures: CANN error 107033 and a silent, no-progress stall.

The investigation separated two mechanisms. Encoder and decoder graph-update transactions could contend on the same update stream and violate the CANN task-group state contract. Independently, a cache-hit H2D transfer issued from a request-building thread could enter a shared Host submission queue and, under specific memory and timing conditions, form a dependency cycle with decoder compute and the encoder update signal.

The final fix does not rely on additional global locking or thread-scheduling luck. It makes ownership explicit: the encoder graph runner owns FIA graph updates, the encoder worker owns NPU cached-embedding transfers, and admission completes only when an embedding is actually ready for LM consumption.

## 1. Background

The target configuration combines:

- encoder graphs;
- breakable prefill graphs;
- full decode graphs;
- concurrent requests;
- both cache-hit and cache-miss audio admission paths.

A graph normally captures a device computation once and replays it against static buffers, reducing Python scheduling and operator-submission overhead.

The speech encoder adds another form of dynamism. Two inputs can have the same total token count and reuse the same graph bucket while carrying different audio-window layouts. Each Ascend fused-infer-attention (FIA) task captures host-side cumulative sequence boundaries, so every replay must update every FIA layer with the boundaries for the current input.

The simplified handshake is:

```text
prepare the current encoder input
  → enqueue graph replay
  → replay reaches the event wait before FIA
  → update FIA parameters on an update stream
  → record the completion event
  → continue replay with the current boundaries
```

The same service process also performs decoder graph replay and task updates. Its execution path includes multiple device streams, the torch_npu Host Repository, CPU-to-NPU transfers after cache hits, and futures that control request admission. The failure emerged at the boundary between these mechanisms rather than inside one isolated operator.

The main runtime investigated here used Python 3.11.10, PyTorch 2.10.0, torch_npu 2.10.0.post2, CANN 9.0.1, and unmodified SGLang v0.5.19 at `0bcd8223`.

## 2. Two Failures That Must Not Be Conflated

### 2.1 CANN 107033

One failure reported:

```text
ACL_ERROR_STREAM_TASK_GROUP_STATUS
Unable to start update tasks because the stream is busy
```

The version-matched CANN source shows that `StreamBeginTaskUpdate` requires the stream task-group state to be `NONE`, changes it to `UPDATE`, and relies on `StreamEndTaskUpdate` to restore it.

The following interleaving is therefore invalid:

```text
Encoder: BeginTaskUpdate
Decoder: BeginTaskUpdate   ← the stream is still in UPDATE, so 107033 is returned
Encoder: EndTaskUpdate
```

CANN's internal mutex protects an individual API call. It does not automatically make the caller's entire `Begin → operations → End` sequence atomic. Historical runs showed a shared update stream and 107033, but the first failure did not record the full identities of both transactions. Transaction overlap is therefore supported by both source and runtime evidence, but the exact pair involved in that first incident was not directly reconstructed.

### 2.2 Silent Stall

The second failure produced no Python exception and no clear CANN error. Requests stopped completing, threads remained around graph-update submission, and an external watchdog eventually had to terminate the process.

Later evidence showed that this was not simply another manifestation of 107033. It involved a dependency cycle across the Host submission queue, an implicit H2D synchronization, and graph completion events.

## 3. Hypotheses, Discriminating Tests, and Rejections

| Hypothesis | Discriminating check | Conclusion |
| --- | --- | --- |
| Encoder and decoder update transactions interleave on one update stream | Compare failure site, stream identities, and CANN source | Explains 107033, but not the later silent stall |
| The main problem is decoder helper-thread scheduling | Move decoder update submission to the caller thread | Several early rounds passed, but warmup and cold concurrent runs failed later; timing changed, ownership did not |
| One shared submission lock is sufficient | Lock wait, replay, Begin/End, and event record | Reduces transaction-overlap risk, but the full concurrent gate can still stall |
| A private encoder device stream provides isolation | Give each encoder runner its own update stream | The stall remains; separate device streams do not imply separate Host queues |
| torch_npu lost an EventFD wakeup | Inspect queue state, thread stacks, and double-check logic | Direct evidence was missing, and the source did not support a simple lost-wakeup claim |
| `non_blocking=True` guarantees that H2D cannot block the Host | Audit torch_npu/CANN copy paths and run controlled probes | False; pageable Host memory may still cause synchronous waiting |

These negative results mattered. A few passing stress rounds only show that one schedule did not trigger the bug; they do not prove that the modified thread was the defect owner.

## 4. Source Review Corrected the Mental Model

### 4.1 Device streams are not Host submission queues

Version-matched torch_npu source showed that, in the relevant queue mode, work for distinct NPU streams could still enter the same default Host Repository. Some stream conversion and graph-update operations drain the relevant Repository before returning the raw stream.

Therefore:

```text
different device streams ≠ different Host submission queues
```

This is why assigning a private device stream to the encoder was not sufficient on its own.

### 4.2 `non_blocking=True` is not an absolute runtime guarantee

Cached audio embeddings live on CPU. The old path called `.to(device, non_blocking=True)` from a request-building thread.

That flag describes the requested API behavior, not every lower-level condition. For pageable or unregistered Host memory, the runtime can still synchronize the target stream inside the copy path. The Python call may look asynchronous while the Host Repository consumer is blocked in a stream synchronization.

### 4.3 The future completed under the wrong contract

The old implementation attached the cached embedding and immediately returned a completed future. In practice, the future meant "the cache entry was found and a copy was initiated." Its consumer interpreted it as "the embedding is ready for safe LM consumption."

Device completion and request admission had become semantically misaligned.

## 5. The Silent-Stall Dependency Cycle

A repaired diagnostic collector obtained complete thread stacks. Combined with version-matched torch_npu/CANN source and a reduced probe that preserved the same execution mechanism, the dependency chain was:

```mermaid
flowchart TD
    A[Decoder compute replay] -->|waits for encoder update signal| B[Graph update submission]
    B -->|waits for shared Host Repository to drain| C[Host Repository consumer]
    C -->|executes cache-hit H2D callback| D[Implicit synchronization for pageable H2D]
    D -->|waits for decoder compute to finish| A
```

In short:

```text
compute waits for update
update waits for the Host queue
H2D work in the Host queue waits for compute
```

Controlled probes distinguished the relevant conditions:

- overlap with an ordinary pageable source reproduced the stall;
- removing the overlap completed;
- a pinned-source control completed;
- output checks passed, ruling out a simple data-corruption explanation.

The thread evidence placed the Repository consumer in the implicit H2D synchronization path, the copy submitter waiting for Repository capacity, and the graph updater waiting before stream acquisition or Repository drain.

There is still an evidence boundary: the exact final device task associated with the stalled event was not read directly from the device. The last signal relationship is partly code-derived from the reduced single-event program and API contracts. The evidence closes the causal loop for the reproduced silent-stall mechanism, but it does not prove that every historical failure had the same cause.

## 6. Final Design: Make Device-Work Ownership Explicit

### 6.1 Encoder-owned graph replay and FIA updates

Each encoder graph runner owns an update stream on the same device. During capture, the graph places an external event wait around every updatable FIA task. At replay time, the encoder worker performs one owned handshake:

```mermaid
sequenceDiagram
    participant EW as Encoder worker
    participant ES as Encoder compute stream
    participant G as Captured encoder graph
    participant US as Runner-owned update stream

    EW->>ES: Copy current hidden states into static input
    EW->>US: wait_stream(encoder compute stream)
    EW->>ES: Enqueue graph replay
    ES->>G: Replay reaches captured event.wait
    G-->>ES: Wait for current FIA parameters
    EW->>US: graph_task_update_begin(handle)
    EW->>US: Submit current cumulative window lengths
    EW->>US: graph_task_update_end()
    EW->>US: event.record()
    US-->>G: Release event wait
    G-->>ES: Continue with current boundaries
```

The initial `wait_stream` is a dependency edge, not a device-wide synchronization. It prevents the update sequence from overtaking encoder input preparation while preserving asynchronous graph execution.

Replay is intentionally submitted before the updates. The captured graph already waits before each FIA task; replay establishes the consumer side of the handshake, and the subsequent event record releases it with the current parameters.

The graph handle, update stream, task events, and failure state now share one owner and lifetime. If replay, update Begin/End, the FIA operation, or event recording fails after the handshake starts, a retry may leave a graph waiting for a signal that will never arrive. Such failures are terminal for the encoder worker. Allocation, eager warmup, pool acquisition, and backend-setup failures before graph-context entry leave only the affected bucket on the eager path.

### 6.2 Encoder-worker-owned cached embedding transfer

An NPU cache hit no longer submits H2D from the request builder. It becomes a `CachedEmbeddingTransfer` and enters the existing single encoder-worker queue:

```mermaid
flowchart LR
    A[Request builder finds CPU cache hit] --> B[submit_cached_embedding]
    B --> C[Enqueue CachedEmbeddingTransfer]
    C --> D[Encoder worker / execute_batch]
    D --> E[Enter encoder private stream]
    E --> F[Submit CPU-to-NPU copy]
    F --> G[record_stream for LM consumer lifetime]
    G --> H[Synchronize encoder stream]
    H --> I[Resolve transfer future]
    I --> J[DeferredAdmission releases request]

    B -. non-NPU .-> K[Caller-thread attach]
    K --> L[Already-completed future]
    L --> M[Immediate admission]
```

The completion contract changes from:

```text
old: future ready = cache hit found and copy attempted
new: future ready = embedding is safe for LM consumption
```

The worker retains the CPU source until synchronization so cache eviction cannot invalidate an in-flight copy. It registers the LM consumer stream with `record_stream()` for allocator safety and synchronizes the transfer stream before resolving the future. `record_stream()` protects storage lifetime; it does not by itself prove that H2D has completed, so it cannot replace synchronization at the admission boundary.

A transfer failure rejects admission. Single-flight followers reuse the leader's completed device work and publish only ready embedding metadata; they do not issue duplicate device work from callback threads.

### 6.3 Ownership and completion contract

| Component | Owns | Completion signal |
| --- | --- | --- |
| Request-building thread | Request construction and deferred admission | Does not submit NPU cached-transfer work |
| Encoder worker | Encoder batches and NPU cache-hit transfers | Resolves the future only after stream synchronization |
| Encoder compute stream | Static-input copy, encoder graph replay, and transfer work | Ordered by stream dependencies |
| Per-runner update stream | FIA task-group updates and completion events | Event records release graph replay |
| LM/default stream | Consumption of the ready embedding | Registered with `record_stream()` for allocator lifetime |

CUDA, ROCm, XPU, and CPU keep the previous caller-thread attach and already-completed future. They do not enter the NPU transfer queue, and their existing graph and eager-fallback behavior remains unchanged.

## 7. Why No SGLang Patch Is Required

The final change corrects ownership and admission inside SGLang-Omni:

- the encoder runner owns the full NPU graph-update lifecycle;
- the encoder worker owns cache-hit H2D;
- the request builder no longer injects NPU copies into a shared submission path;
- requests enter the LM only after the embedding is actually ready.

The design therefore works with unmodified SGLang v0.5.19 at `0bcd8223` and does not depend on SGLang PR #40059.

## 8. Validation and Evidence Boundaries

### Completed qualification

Native CPU regression on the qualified candidate reported:

```text
562 passed, 1 guarded XPU skip, 2 accelerator deselected
```

One cold-start, concurrency-8, 140-input all-graph run completed with 140/140 successful HTTP responses, no failures, empty transcripts, or high-WER samples, corpus WER `0.011373`, and zero per-sample disagreement against the retained reference. Decode graph runtime counters reported no eager fallback, and shutdown and resource cleanup completed normally.

A subsequent 20-round independent cold-start stability run completed 2800/2800 HTTP 200 responses. Every round produced WER `0.011373`; reference disagreement, decode fallback, watchdog triggers, and forced shutdowns were all zero.

After conflict resolution, merge head `b8ea8032` completed native CPU regression and another 3/3 independent cold-start rounds: 420/420 HTTP 200 responses, unchanged WER, zero decode fallback, zero watchdog triggers, and normal cleanup.

### Current head must not inherit older hardware evidence

At publication preparation time, the PR's squashed head is `f560a3c7`. It adds review cleanup that narrows terminal capture-failure classification, scopes worker-owned cached transfers to NPU, and clarifies the metadata-only helper name.

The current head has source-isolated CPU checks, compile checks, targeted pre-commit checks, and a full diff audit. The hardware results above directly qualify `555c6906` and `b8ea8032`, not `f560a3c7`. Refer to [PR #2160](https://github.com/sgl-project/sglang-omni/pull/2160) for the latest Validation section and checklist.

Additional boundaries:

- server results were operator-reported; retained raw artifacts were not independently imported and recomputed locally;
- update-stream ownership is supported by pinned source and tests, but runtime stream handles were not directly sampled;
- encoder capture markers are not per-replay counters, so exact encoder replay counts remain unmeasured;
- the results cover fixed versions, configuration, and the SeedTTS EN workload, not every Ascend deployment.

## 9. Engineering Lessons

1. **Passing a few stress rounds does not establish root cause.** Moving work between threads, adding a lock, or changing streams may only alter failure probability.
2. **Device streams and Host submission queues must be analyzed separately.** Isolation of one does not imply isolation of the other.
3. **`non_blocking=True` is not an absolute asynchronous guarantee.** Host-memory type and runtime implementation still matter.
4. **A future's completion contract must match downstream usability.** "Submitted" and "ready to consume" are different states.
5. **A durable fix removes the invalid dependency instead of accumulating synchronization.** The key was reclaiming device-work ownership, not adding another lock.
6. **Error codes and silent stalls need separate evidence chains.** They share an ownership problem, but they are not one failure.
7. **Implementation, tests, and documentation should express the same completion contract.** The component that submits device work should own its lifetime, completion signal, and failure handling.

The redesign ultimately clarified three boundaries:

```text
who owns device submission
who decides when a request may advance
who owns the complete graph-update transaction lifecycle
```

Once those boundaries became explicit, correctness stopped depending on favorable scheduling and became a set of explainable dependencies and testable failure semantics.
