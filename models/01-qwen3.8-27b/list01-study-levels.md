# Qwen3.8-27B — a gentle start in inference engineering

One guide, from your first serving question to a measured deployment decision. **Start with the first session below. Open the detailed steps only when you reach that stage.** You do not need to read the whole document before beginning.

Updated and primary sources rechecked: **2026-09-13**. This is a study plan; all tasks and gates are uncompleted.

## Your first session — one question and your GPU

**Learn:** inference means running a trained model to produce an answer. Serving adds the practical work around it: receiving requests, keeping model state, scheduling GPU work and returning answers.

**Read:** only the [official model card's Model Overview](https://huggingface.co/Qwen/Qwen3.8-27B#model-overview). Leave benchmark scores and model internals for later.

**Do:** write one serving question in your own words, choose a small text-only workload and a provisional acceptable latency and answer check, then record the NVIDIA GPU you already have access to: its model, usable VRAM and permissions. These are **Q38-L01-A01/A02**, detailed in Stage 1. Stop this session with a question and a short inventory. No deployment choice is needed yet.

The next session is Stage 2: identify the model and predict whether a first configuration fits. Ask for help after your prediction or first attempt. A direct worked explanation is available when requested; keep it distinct from independent evidence.

## The path through this guide

| Where | The question you will answer | What to open now |
| --- | --- | --- |
| **Intro, Stages 1–3** | What am I serving, and can I run it correctly? | One stage at a time |
| **Intro, Stages 4–7** | What changes when prompts, answers or load grow? | After the endpoint works |
| **Medium, Stages 8–17** | Can I diagnose, improve and defend a useful service? | After Intro evidence, or by a deliberate study choice |
| **Optional extensions** | Would precision, reuse, two GPUs or Kubernetes answer a real question? | Only the extension you choose |
| **Expert return path** | Does one measured problem warrant implementation or distributed depth? | Keep closed until chosen |

The normal stopping point is **Intro → Medium core → the next chosen model**. GLM-4.7-Flash is the current next model; the order remains your choice. Expert is deferred. Moving on with a named gap is allowed; it does not demonstrate the missing capability.

There are exactly two primary lists per level: **A = learn/read/do work**, **B = evidence checks**. Stages are readable groups within A. The old Q38 and practical L1/L2/L3 IDs are retained inside their stages, so existing references keep their meaning. Overlapping checks share evidence; they do not require duplicate experiments.

The first warm baseline is a useful milestone. Full Intro also includes prompt, output and modest load comparisons. Medium includes two isolated tuning mechanisms and a trace/source-path investigation. These fuller requirements already existed in the source plans.

## Words you will meet, explained briefly

<details markdown="1">
<summary>Open a plain-language glossary when you meet a new term</summary>

| Word | Meaning in a serving experiment |
| --- | --- |
| Token | A unit of text produced by the tokenizer; count the actual templated input and generated output. |
| Checkpoint / revision | The trained model files / the exact version of those files. |
| Runtime / engine | Software that loads the model and schedules inference, such as vLLM. |
| GPU / VRAM / HBM | The accelerator / its device memory / high-bandwidth device memory. Capacity and bandwidth answer different questions. |
| Weights | The learned tensors loaded for inference. |
| Prefill / decode | Processing the input prompt / generating the continuation step by step. |
| KV cache | Saved attention keys and values from earlier tokens, reused during generation. |
| Recurrent state | A running summary updated by the linear-attention layers; the hybrid model also has growing attention KV. |
| BF16 / quantization | A 16-bit numeric format / using a reduced-precision representation. Runtime support and answer quality still need checks. |
| Latency / throughput | How long an individual request takes / how much work the service completes per second. |
| Concurrency / arrival rate | Requests active together / new requests offered per second. These are different load controls. |
| SLO / goodput | A declared service target / useful requests meeting that target per second. |
| TTFT / TPOT | Time to first generated output / an average time-per-output-token estimate. See the measurement reference before using formulas. |
| Trace / kernel | A timeline of CPU/GPU activity / a GPU operation implementation. |
| Replica / tensor parallelism (TP) | An independent model copy / splitting model computation across GPUs. |
| Prefix reuse / speculation / MTP | Reusing work for a shared beginning / drafting then verifying tokens / a trained multi-token prediction component. |
| Topology / NUMA / NIC | How devices connect / CPU-memory locality / a network adapter. |
| Collective / NCCL / RDMA | Coordinated rank-to-rank communication / NVIDIA's collective library / a direct-memory network transport. |
| Disaggregation | Separating prefill and decode workers, which requires correct transfer of all necessary model state. |

These are orientation definitions. Each stage links the source needed for its actual experiment.

</details>

## A few model facts — use them when you reach Stage 2

<details markdown="1">
<summary>Open when you reach Stage 2 — model identity and support</summary>

Qwen3.8-27B has a dense language backbone and a vision encoder. Its language stack mixes **48 Gated DeltaNet layers with 16 full-attention layers**. Dense FFNs and hybrid attention describe different parts of the block. The pinned configuration still names its implementation `qwen3_5`; exact artifacts determine support. [Checkpoint configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json).

The model card describes thinking enabled by default, reasoning-effort controls, preserved thinking history, native context of 262,144 tokens and MTP. Treat these as model capabilities to verify in the selected runtime. Begin with a smaller text-only envelope. Loading vision or MTP components and exercising them are separate facts. [Official model card](https://huggingface.co/Qwen/Qwen3.8-27B).

Recipes are starting references, not a guarantee of fit or speed on your GPU. Pin actual versions before a run. Use vLLM first; selected SGLang reproductions and Dynamo routing/disaggregation can follow a concrete question. Generic state-transfer documentation does not prove complete Qwen hybrid-state support. [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), [SGLang recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B), [Dynamo routing](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing), [Dynamo disaggregation](https://docs.nvidia.com/dynamo/cli/disaggregated-serving/overview).

</details>

## How small is a step?

<details markdown="1">
<summary>Open for timeboxes, hardware labels and evidence rules</summary>

Aim for a **5–20 minute reading/preparation action**, or a **15–45 minute connected task**. An experiment can span several sessions. Split it into prediction/setup, a bounded run, analysis, then explanation. GPU run time, downloads and access problems are separate unknowns. The old plans' 45–90 minute preparation, 3–5 hour Intro and 8–16 hour Medium estimates were rough budgeting ideas; the merged evidence requirements determine the work, so these are not total-duration promises or deadlines.

**Core** is required for the level's B checks. **Extension** is optional unless selected beforehand as part of a required comparison. **Later** is deferred. A rejected optimization or no-speedup result can be useful evidence.

Hardware labels in the detailed steps: **P0** = source/paper work; **CPU** = executable host work; **G1** = one suitable NVIDIA GPU; **G2** = multiple GPUs on one host; **N2** = at least two suitable hosts with recorded fabric/topology. A calculation, provider API, CPU toy or borrowed trace does not establish target-GPU or fabric measurements.

Use the existing NVIDIA access first; model, VRAM, topology and privileges still need recording. The initial 80 GB-class/BF16 suggestion in the practical checklist is a possible reference configuration, not a hardware requirement. A documented quantized route may be appropriate after inventory. Rental budgeting steps apply only if you later choose authorized paid compute. Shared-service changes, rentals and uncontrolled stress tests are outside this planning task.

Unsupported features can be recorded as **N/A — reason**, and unavailable resources as **BLOCKED — requirement**. Keep the gap visible. Scores, assistance records and evaluator judgments stay in existing private records; this public plan is not a completion or readiness ledger.

</details>

## level01-intro — run correctly, then make small measurements

Scope: source-bound architecture and memory reasoning, one correct text-only target-model deployment, and bounded baseline/length/load observations. Vision, custom kernels and distributed deployment can wait.

### A — Learn, read and do

#### Stage 1 — Choose a serving question and inspect your GPU

**Core · P0 → CPU/G1 · Before this: none.**

**Learn:** the difference between an advertised model capability and your lab target.

**Read:** [model card: Model Overview](https://huggingface.co/Qwen/Qwen3.8-27B#model-overview); your host GPU and access information.

**Do next:**

1. Write the serving question, workload limits, acceptable latency and a checkable answer rule.
2. Record GPU/VRAM, CPU/RAM, driver/runtime, topology and deployment/profiling permissions.
3. Save a short scope and environment note. Mark unknown fields.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L01-A01

**core · P0 · none**

State my first serving question and freeze a text-only Intro scope. Pick a small input/output envelope, task-correctness rule and latency target I can test.

**Save and check:** A short experiment contract; separate chosen targets from facts about the model.

##### Q38-L01-A02

**core · CPU/G1 · A01**

Inventory the accessible GPU, VRAM, driver/toolkit, host RAM, CPU, device topology and permissions. Identify whether this is an isolated lab or shared resource.

**Save and check:** Environment manifest without credentials; list unavailable profiling or deployment privileges rather than guessing.

</details>

#### Stage 2 — Identify the model and predict memory fit

**Core · P0/CPU · Before this: Stage 1.**

**Learn:** weights versus runtime memory; growing KV versus recurrent state; how the hybrid block is arranged.

**Read:** [model card: Model Overview](https://huggingface.co/Qwen/Qwen3.8-27B#model-overview), [pinned checkpoint configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json); [vLLM recipe: prerequisites and the launch section for your GPU](https://recipes.vllm.ai/Qwen/Qwen3.8-27B). Open the memory reference below only for the calculation you are doing.

**Do next:**

1. Pin the model/runtime and draw one repeated block with its state.
2. Estimate weights and two KV cases; add recurrent/convolution state and unknown overhead rows.
3. Choose a conservative first configuration from your GPU inventory. Save an executable worksheet and the source-bound ledger.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L01-A03

**core · CPU · A02**

Pin the exact model repository/revision and a supported runtime version. Inspect the model config, tokenizer/processor and runtime recipe.

**Save and check:** Artifact/version manifest with primary links; identify model and implementation names, supported dtypes and unresolved support questions.

##### Q38-L01-A04

**core · P0 · A03**

Annotate one repeated block in the model map: token mixing, normalization, FFN, residuals, attention positional encoding, and where vision/MTP are omitted. Count layer types from config.

**Save and check:** My own annotated execution path; every numerical geometry claim is config-bound, with any diagram disagreement noted.

##### Q38-L01-A05

**core · CPU · A03**

Predict weight storage from actual tensor count and selected bytes per element. Convert GB/GiB and separate checkpoint size from GPU runtime allocation.

**Save and check:** A small executable memory worksheet; quantify omitted metadata, scales and runtime overhead. Do not turn the 27B label into a full-capacity prediction.

##### Q38-L01-A06

**core · CPU · A04/A05**

Derive full-attention KV growth from the actual KV heads, head dimension, layer count, cache dtype and live tokens. Predict two context/concurrency cases.

**Save and check:** Unit-checked KV worksheet; distinguish token-linear KV from fixed-size recurrent state and from allocator reservation.

##### Q38-L01-A07

**core · P0/CPU · A03/A06**

Locate linear-attention recurrent and convolution state shapes and runtime slot/checkpoint handling. Account for state dtype and possible multiple slots per request.

**Save and check:** A source-bound state ledger; explicitly leave runtime-dependent unknowns unresolved until startup logs or code confirm them.

##### Practical steps — L1-A

**Consequence:** prevents applying the wrong memory or parallelism model.

- [ ] **L1-A01** Save the checkpoint identifier and resolve its actual revision SHA. Record tokenizer/processor revisions if different.
- [ ] **L1-A02** Locate `layer_types`, `num_key_value_heads`, `head_dim` and `dtype` in the configuration. Annotate their serving implications in one sentence each.
- [ ] **L1-A03** Explain the distinction between dense FFNs and hybrid attention. State why expert parallelism is not the natural scaling mechanism for this backbone.
- [ ] **L1-A04** List the state kept by full attention versus DeltaNet. State that a fixed recurrent-state shape does not make the entire hybrid model constant-memory with context.

**Evidence:** a short model fact sheet. **Stop:** you can explain the request's state without deriving DeltaNet.

##### Practical steps — L1-B

**Consequence:** avoids choosing a GPU from parameter count alone.

- [ ] **L1-B01** Reproduce the BF16 weight estimate in the Stage 2 memory reference below. Distinguish decimal GB from binary GiB.
- [ ] **L1-B02** Reproduce the KV calculation and the four-sequence example. Explain why KV heads, rather than query heads, enter that formula.
- [ ] **L1-B03** Sketch a memory budget with separate rows for weights, attention cache, recurrent state, graph/workspace allocations and safety headroom. Mark unknown rows explicitly.
- [ ] **L1-B04** Compare idealized 16-, 8- and 4-bit weight payloads. Explain why scale metadata, mixed precision and kernel support prevent assuming exact halving of runtime memory or latency.

**Evidence:** one small memory table, including assumptions. **Stop:** choose a feasible first configuration; do not demand an exact allocator prediction.

##### Practical steps — L1-C

**Consequence:** makes failures and measurements interpretable.

- [ ] **L1-C01** Choose one NVIDIA GPU, one serving engine and one precision. Preferred reference: a sufficiently provisioned 80 GB-class GPU with BF16. If cost/capacity dictates otherwise, use a documented compatible quantized checkpoint and label it as the baseline.
- [ ] **L1-C02** Start with text-only requests, TP=1, concurrency=1, maximum context around 16,384, no speculative decoding and no weight offload. Explicitly record cache behaviour and whether the runtime still loads vision/MTP components.
- [ ] **L1-C03** Define a rental spend cap, storage allowance and shutdown point. Include download/startup/idle time in the estimate; stop the GPU while doing lengthy reading.
- [ ] **L1-C04** Write a configuration manifest: GPU/SKU/VRAM, CPU/RAM, driver, CUDA runtime, container digest, engine version, checkpoint revision, dtypes, launch arguments and client location. Unknown hardware details remain unknown.

**Evidence:** a completed configuration manifest. **Stop:** no need to select the cheapest GPU in the market or install Kubernetes.

Split this stage into model identity, block annotation, weight worksheet, KV worksheet, then recurrent-state ledger. A first fit estimate can contain explicit unknowns; reconcile them after startup. Deriving the DeltaNet update equations is not a prerequisite.

</details>

#### Stage 3 — Get a correct, repeatable response

**Core · CPU → G1 · Before this: Stage 2.**

**Learn:** endpoint readiness, chat templates, stop behavior and reasoning/final-answer separation.

**Read:** [vLLM recipe: prerequisites and the launch section for your GPU](https://recipes.vllm.ai/Qwen/Qwen3.8-27B); [model card: API usage](https://huggingface.co/Qwen/Qwen3.8-27B#api-usage).

**Do next:**

1. Save a tiny fixture set: short/long prompt, bounded output, multi-turn, stop/cancellation boundary and a checkable task.
2. Launch the supported target configuration and record acquisition/loading/compilation/readiness plus memory limits.
3. Run fixture correctness checks and save the launch configuration. Keep the endpoint private or use an authenticated access path.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L01-A08

**core · CPU · A01/A03**

Build a tiny text fixture set: short request, longer prompt, controlled output budget, multi-turn chat, stop/cancellation boundary, and one checkable task.

**Save and check:** Saved fixtures and validation rules; use deterministic input fixtures, not a universal promise of byte-identical stochastic outputs.

##### Q38-L01-A09

**core · G1 · A02–A08**

Start the target model with a conservative supported configuration that fits the inventory. Choose a documented precision if BF16 is unsuitable and label it. Record startup memory and admitted limits.

**Save and check:** Launch/configuration and startup report; no unapproved shared-service changes. A fit failure remains evidence to diagnose, not a successful deployment.

##### Q38-L01-A10

**core · G1 · A08/A09**

Check the actual chat template, assistant response, stop handling, reasoning/final output separation and multi-turn behavior.

**Save and check:** Correctness results for the saved fixtures; show that the integration is operating as intended before timing it.

##### Practical steps — L1-D

**Consequence:** establishes the minimum operational skill behind inference work.

- [ ] **L1-D01** Confirm GPU visibility inside the runtime and record available VRAM. Check that driver/runtime compatibility and the model backend match the selected recipe.
- [ ] **L1-D02** Download/load the checkpoint using a persistent cache. Separate weight acquisition, model loading, compilation and readiness times where logs permit.
- [ ] **L1-D03** Start the server, retain startup logs and send one short text request. Check the returned model identity, output, finish reason and token usage.
- [ ] **L1-D04** Set thinking explicitly for the request and verify the observed response. Use a private endpoint or authenticated access path; record timeouts and the streaming setting.

**Evidence:** launch command, successful response and startup log. **Stop:** you have a working endpoint and know how to start it again.

</details>

#### Stage 4 — Capture the first useful baseline

**Core · CPU/G1 · Before this: Stage 3.**

**Learn:** cold startup versus warmed requests; client timing versus engine timing.

**Read:** [vLLM benchmark options](https://docs.vllm.ai/en/latest/cli/bench/serve/); the metric definitions in the measurement reference below.

**Do next:**

1. Define send, first non-empty generated output, first final-answer content if observable, and completion timestamps.
2. Warm consistently; run E00 at about 512 input/128 output tokens, concurrency 1. Save actual counts and ten smoke requests.
3. Repeat a batch and keep startup, first request, warm timings, memory and errors separate.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L01-A11

**core · G1 · A09/A10**

Measure startup, first-request and warmed low-concurrency request behavior separately.

**Save and check:** Cold/warm timing table with a defined warm-up; avoid reporting initialization cost as steady-state decode performance.

##### Q38-L01-A12

**core · CPU/G1 · A10/A11**

Capture client-side time to first token, output count, total latency and an appropriate steady-output metric. Record errors and units.

**Save and check:** Baseline table plus measurement definitions; account for streaming granularity and distinguish client timings from engine metrics.

##### Practical steps — L1-E

**Consequence:** avoids confusing network chunks, reasoning and GPU decode speed.

- [ ] **L1-E01** Record send time, first non-empty generated-content arrival and completion time using a monotonic clock. Do not count a role-only or empty initial chunk as the first generated token.
- [ ] **L1-E02** Distinguish first generated output from first final-answer content when thinking is enabled. If the API hides reasoning tokens, record the observability limit.
- [ ] **L1-E03** Record actual input/output tokens and distinguish per-request output rate from aggregate output throughput. Label chunk gaps separately from true inter-token latency if chunks contain multiple tokens.
- [ ] **L1-E04** Compare the benchmark tool's metric definitions with your own. Use the engine's documented benchmark utility where practical; save its version and arguments.

**Evidence:** a metric dictionary and one raw per-request record. See [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/).

##### Practical steps — L1-F

**Consequence:** creates the reference for every later claim.

- [ ] **L1-F01** Warm the engine with several requests. Keep cold-start timing separate and record whether compilation or graph capture continues during measurement.
- [ ] **L1-F02** Run a small fixed workload, approximately 512 input tokens and 128 output tokens, concurrency 1. Save actual lengths; a maximum-output limit does not force that length.
- [ ] **L1-F03** Repeat ten requests for a first smoke baseline. Record successful completions, TTFT, completion latency, output rate, GPU memory and obvious errors. Do not present ten requests as a reliable tail-latency study.
- [ ] **L1-F04** Rerun one batch to detect a major warm-up or environmental effect. If results differ, annotate the likely cause and uncertainty.

**Evidence:** E00 raw results and a compact summary. **Stop:** the baseline is credible enough to compare experiments, even if it is slow.

This is the first-baseline milestone. It is enough to start asking useful performance questions. Full Intro evidence also includes Stages 5–7; the old G1 smoke checklist appears in the B section as a milestone, not a replacement gate.

</details>

#### Stage 5 — Change prompt length, then answer length

**Core · G1 · Before this: Stage 4.**

**Learn:** prefill and decode respond differently to workload shape.

**Read:** [vLLM benchmark options](https://docs.vllm.ai/en/latest/cli/bench/serve/); E01/E02 shapes in the experiment reference.

**Do next:**

1. Predict what a longer input changes; run E01 with output policy/load held fixed.
2. In a separate comparison, predict and vary output budget for E02 with prompt/load held fixed.
3. Save two small tables or curves with actual generated lengths, repetitions and confounders.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L01-A13

**core · G1 · A12**

Change only input length for a small prefill-focused comparison while holding output policy and load constant. Predict the result first.

**Save and check:** Repeated input-length comparison; explain what is actually isolated and what still confounds the result.

##### Q38-L01-A14

**core · G1 · A12**

Change output budget for a decode-focused comparison while holding prompt fixtures and load constant. Record actual generated length.

**Save and check:** Output-length comparison; explain why output limits, early stops and reasoning tokens affect completed-task latency.

##### Practical steps — L2-A

**Consequence:** reveals which workloads strain prefill or decode.

- [ ] **L2-A01** Predict the effect of increasing prompt length at fixed concurrency and fixed generated length.
- [ ] **L2-A02** Run E01 at roughly 512, 2,048 and 8,192 input tokens with 128 generated tokens. Control prefix reuse; record actual lengths.
- [ ] **L2-A03** Run E02 at roughly 128, 512 and 1,024 generated tokens with a fixed 2,048-token input. Use a controlled generation-length facility for synthetic throughput tests if supported; keep natural stopping for quality tests.
- [ ] **L2-A04** Plot or tabulate TTFT, completion time and output rate against input/output length. Separate engine measurements from client observations.
- [ ] **L2-A05** Explain the observed direction using compute, memory traffic and state access. Treat “prefill is compute-bound; decode is bandwidth-bound” as a hypothesis that depends on workload and hardware.

**Evidence:** two small curves/tables and a paragraph explaining exceptions. **Stop:** you can predict the next workload point qualitatively.

The legacy checklist placed these steps in Medium. Here its IDs are preserved in Intro because the original Q38 Intro already required these comparisons. Reuse this evidence later instead of repeating it automatically.

</details>

#### Stage 6 — Try modest load and sketch the request path

**Core · CPU/G1 · Before this: Stage 5.**

**Learn:** throughput can rise while individual requests slow; utilization alone is not a diagnosis.

**Read:** [vLLM engine metrics](https://docs.vllm.ai/en/latest/usage/metrics/); your topology output.

**Do next:**

1. Try a small concurrency sweep inside the recorded limits; save latency, throughput, memory and errors.
2. Correlate one slow condition with host/GPU observations and sketch client → network → queue → CPU → GPU → stream.
3. Record topology, compare allocated memory with your ledger, restart and repeat the smoke request.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L01-A15

**core · G1 · A06/A07/A12**

Try a modest concurrency sweep within the safe envelope. Collect throughput, request latency, memory and errors with the same request mix.

**Save and check:** A small capacity curve; identify the first observed tradeoff or limit without deliberately exhausting a shared GPU.

##### Q38-L01-A16

**core · CPU/G1 · A13–A15**

Capture host/GPU utilization and memory alongside one slow condition. Sketch client → queue → CPU work → GPU execution. State a bottleneck hypothesis and its alternative.

**Save and check:** A correlated observation plus a path diagram; do not call utilization alone proof of compute, memory or network saturation.

##### Practical steps — L1-G

**Consequence:** starts the networking thread immediately without forcing distributed inference.

- [ ] **L1-G01** Sketch the path: client, network/proxy, API process, tokenizer/scheduler, GPU, response stream. Mark which parts you can observe.
- [ ] **L1-G02** Capture `nvidia-smi topo -m` if available. Identify PCIe/NVLink indications and CPU/NUMA affinity; on a one-GPU instance, explicitly record that GPU collective behaviour was not tested.
- [ ] **L1-G03** Compare estimated weights/cache with startup allocations. Explain at least the major categories of the gap; a preallocated cache pool will not track live tokens linearly in `nvidia-smi`.
- [ ] **L1-G04** Stop and restart the service from the saved configuration, then complete the same smoke request. Save a brief unresolved-questions list.

**Evidence:** request-path sketch, topology capture and restart note.

</details>

#### Stage 7 — Explain what you actually found

**Core · P0 · Before this: Stages 2–6.**

**Learn:** a measured observation, its possible mechanism and the limit of the claim.

**Read:** your own block annotation, memory ledger and E00–E03 evidence.

**Do next:**

1. Explain the architecture/state story without hints.
2. Apply it to a changed prompt/load scenario.
3. Defend the evidence, an alternative bottleneck explanation and the next measurement that would distinguish them.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L01-A17

**core · P0 · A04–A16**

Explain the architecture/state story, apply it to a changed prompt/load scenario, and defend the measurements and limits without AI hints.

**Save and check:** My unaided explanation/application/defense in private practice, plus a supported technical takeaway in my own notes when I choose to write it.

</details>

### B — Evidence needed for Intro

<details markdown="1">
<summary>Check Intro evidence after Stages 1–7</summary>

Open this when checking the work from Stages 1–7. All core tasks above are covered by these gates. Image task Q38-L01-A18 is an optional extension below.

- [ ] **Q38-L01-B01 — Source-bound scope and architecture.** A01–A04: the manifest identifies exact target/runtime/hardware, and I can trace the hybrid block and its omissions. Unresolved support needed for the baseline is resolved or explicitly blocks completion.
- [ ] **Q38-L01-B02 — Memory/state reasoning.** A05–A07/A09: my worksheet distinguishes weights, attention KV, recurrent/convolution state and runtime reservation; units and assumptions are checked against artifacts and startup observations. I can explain why fitting weights does not establish serving capacity.
- [ ] **Q38-L01-B03 — Correct target-model run.** A08–A10: the actual Qwen checkpoint runs on the recorded GPU/configuration and the declared text fixture checks pass. A surrogate, provider-only API or copied report does not satisfy this gate.
- [ ] **Q38-L01-B04 — Measured baseline and phase sensitivity.** A11–A14: raw records and the stated repetition protocol support cold/warm, input-length and output-length comparisons with understandable metric definitions. I can explain confounders; no fixed tok/s threshold is required.
- [ ] **Q38-L01-B05 — Capacity and plausible diagnosis.** A15/A16: the bounded load comparison records latency, throughput, memory and errors, and I can defend a bottleneck hypothesis plus a test that could disprove it. A diagnosis can remain provisional if the next measurement is specified.
- [ ] **Q38-L01-B06 — Interview perspective.** A17: independently explain the concept, apply it to a changed serving scenario and defend the actual evidence without claiming vision, multi-GPU or kernel expertise. A18 is optional and not an Intro graduation condition.

<details markdown="1">
<summary>Earlier first-baseline milestone — G1 IDs retained</summary>

These checks remain useful after the first baseline and restart. They are a smaller milestone; full Intro uses the Q38-L01-B gates above.

- [ ] **G1-01** I can load the pinned configuration and get a valid response again.
- [ ] **G1-02** I can explain the weight estimate, growing attention cache and additional state.
- [ ] **G1-03** I have a warm baseline with explicit token counts and measurement definitions.
- [ ] **G1-04** I can distinguish client/network delay, queueing, prefill and decode conceptually.
- [ ] **G1-05** I have one performance prediction to test in Level 02.

</details>

</details>

## level02-medium — diagnose, compare and defend a service

Scope: reusable experiments, scheduling plus a distinct memory/admission or supported-precision comparison, trace/source-path diagnosis, correctness, reasoning policy, network attribution, recovery and economics. Start with the existing GPU.

### A — Learn, read and do

<details markdown="1">
<summary>Open Medium Stages 8–17 and optional extensions when ready</summary>

#### Stage 8 — Make the baseline reusable

**Core · CPU/G1 · Before this: Intro core evidence.**

**Learn:** a runner fixes fixtures, settings and timing so comparisons mean something.

**Read:** [vLLM benchmark options](https://docs.vllm.ai/en/latest/cli/bench/serve/); your Intro manifest and raw results.

**Do next:**

1. Freeze the workload, quality/latency budgets and the two comparison mechanisms before running them.
2. Build a small reusable runner with structured results and response validation.
3. Reproduce a baseline and explain material drift before tuning.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A01

**core · CPU · Intro core evidence**

Freeze the Medium workload/SLO/quality envelope and the selected comparison branches. Build a reusable runner with recorded fixtures, settings, response validation and structured result output.

**Save and check:** Executable runner and protocol; explicitly select scheduling plus one distinct memory/precision/admission mechanism for controlled testing.

##### Q38-L02-A02

**core · G1 · A01**

Reproduce the Intro baseline with the runner, then repeat a condition. Check sample counts, spread, warm-up and timing overhead.

**Save and check:** Reproduction report; diagnose a material difference before comparing a tuning change.

</details>

#### Stage 9 — Find useful capacity under different demand

**Core · G1 · Before this: Stage 8.**

**Learn:** closed-loop concurrency differs from offered arrival rate; queues change the user experience.

**Read:** [vLLM engine metrics](https://docs.vllm.ai/en/latest/usage/metrics/); E03 and your input/output curves.

**Do next:**

1. Run a modest concurrency grid and a separate arrival-rate/request-mix grid.
2. Count offered, admitted, completed and failed requests; verify the client can generate demand.
3. Select an operating envelope under your stated latency/quality targets and identify the first limiting resource.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A03

**core · G1 · A02**

Sweep a small arrival-rate/request-mix grid separately from a concurrency-limit sweep. Include long-prefill/short-decode and short-prefill/long-decode cases.

**Save and check:** Latency/throughput/error curves; distinguish generated demand, admitted requests, completed requests and queueing.

##### Practical steps — L2-B

**Consequence:** distinguishes high throughput from an acceptable user experience.

- [ ] **L2-B01** Define a provisional workload-specific SLO before the sweep: acceptable TTFT, time-per-output-token or completion latency, and error rate. Label it a lab target, not a universal service standard.
- [ ] **L2-B02** Sweep concurrency 1, 2, 4, 8 and, only if useful, 16 at a fixed shape. Stop after a clear saturation/latency boundary or memory pressure.
- [ ] **L2-B03** Record aggregate output tokens/sec, request rate, per-request latency, running/waiting requests and cache pressure. More occupied memory or higher GPU utilization alone does not prove efficiency.
- [ ] **L2-B04** At the chosen operating point, run a modest offered-arrival-rate test as well as closed-loop concurrency. Verify the client can generate the requested load and count offered, admitted, completed and failed requests separately.
- [ ] **L2-B05** Name the best operating point for your target and explain what saturates first. Show the point where adding load mainly increases queueing.

**Evidence:** capacity table with SLO pass/fail and a load-generation description. **Stop:** no exhaustive search over every request rate.

</details>

#### Stage 10 — Compare two mechanisms, one at a time

**Core · CPU/G1 · Before this: Stages 8–9.**

**Learn:** scheduler token budgets and memory/admission limits solve different problems.

**Read:** [vLLM tuning: the setting selected for this experiment](https://docs.vllm.ai/en/latest/configuration/optimization/); the runtime pool/slot evidence in your memory ledger.

**Do next:**

1. Compare a scheduling lever with the baseline and two settings; hold other variables fixed.
2. Separately predict and measure one memory/admission change, or a supported precision branch chosen before the comparison.
3. Validate every candidate and save interim adopt/reject results. After Stage 13, return to Q38-L02-A19 to consolidate and retest the changed-workload boundary.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A04

**core · G1 · A03**

Test one documented scheduling lever, such as chunked prefill or batched-token limits, in a pinned runtime. Hold other knobs and workload constant.

**Save and check:** A/B scheduling report; account for throughput and decode/request latency instead of reporting only the faster case.

##### Q38-L02-A05

**core · CPU/G1 · A02/A03**

Refine the state/KV/overhead ledger using runtime pool/slot observations. Predict a supported admission, sequence-limit or memory-pool change and measure its capacity effect.

**Save and check:** Predicted-versus-observed capacity table; reconcile live use and reserved allocation, including hybrid-state slots when applicable.

##### Q38-L02-A07

**core · CPU/G1 · A04/A05; A06 if chosen**

Run the declared quality/correctness fixtures on every selected candidate configuration. Check stop behavior, output truncation, failed requests and task success.

**Save and check:** Quality/error comparison; a faster configuration outside the frozen quality budget is rejected for that workload.

##### Q38-L02-A19

**Return after Stage 13:** consolidation uses the trace from A12. The scheduling and memory comparisons above are the work to do now; this later synthesis reuses them.

**core · CPU/G1 · A04/A05/A07/A12**

Consolidate two controlled changes with different mechanisms: scheduling plus memory/admission or supported precision. Retest the original and a changed workload.

**Save and check:** One comparative report linking existing experiments; explain causality and a no-gain/regression boundary. Separate individual changes before testing a combined configuration.

##### Practical steps — L2-C

**Consequence:** makes tuning deliberate rather than a collection of copied flags.

- [ ] **L2-C01** Read the selected engine's definitions of maximum context, sequence concurrency, token budget and GPU-memory budget. Identify the knob matching your measured bottleneck.
- [ ] **L2-C02** Choose one knob, record a baseline and two alternatives, and predict the latency/throughput/memory tradeoff. For example, change the batched-token budget under a mixed short/long-prompt workload.
- [ ] **L2-C03** Run the controlled comparison without changing precision, GPU, reasoning or workload distribution at the same time. Record effective settings from logs.
- [ ] **L2-C04** Inspect long-prefill interference with ongoing decode, or cache/preemption behaviour if that was the target. Use server metrics where exposed.
- [ ] **L2-C05** Adopt or reject the change. Recheck representative output correctness. If no gain appears, explain why and retain the baseline; investigate a second knob only if a concrete bottleneck remains.

**Evidence:** a before/after decision with the cost of the tradeoff. Reference: [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).

Split this into scheduling setup/run/analysis, memory setup/run/analysis, correctness checks, then changed-workload defense. The old practical G2 required one optimization attempt; the Q38 plan required two distinct mechanisms. Both are retained, with the fuller Q38 gate governing Medium. A supported precision branch can supply the second mechanism; it does not remove the scheduling comparison or memory-ledger work. If that branch is chosen, finish the memory ledger, visit Stage 11 and the weight-precision extension to verify support and run it, then return here. A19 synthesis comes after Stage 13.

</details>

#### Stage 11 — Check optional-feature support before trying flags

**Core · P0/CPU · Before this: Stage 10 memory ledger.**

**Learn:** prefix reuse, state branching, MTP and state transfer need exact model/runtime support.

**Read:** [pinned checkpoint configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json); [vLLM recipe: prerequisites and the launch section for your GPU](https://recipes.vllm.ai/Qwen/Qwen3.8-27B); [hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/).

**Do next:**

1. Make a small source/code-backed matrix for prefix reuse, recurrent-state precision, checkpointing/branching, speculation and complete hybrid-state transfer.
2. Separate supported, locally tested, unknown and unavailable.
3. Choose an extension only when it answers the measured question; leave other branches deferred.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A08

**core · P0/CPU · A05**

Build a support matrix for prefix reuse, recurrent-state precision, branching/checkpointing, MTP/speculation and full hybrid-state transfer on the pinned runtime/model.

**Save and check:** Primary-source/code-backed support matrix; separate supported, tested, unknown and unavailable. An accepted flag is insufficient.

</details>

#### Stage 12 — Measure time and cost to a useful answer

**Core · G1 · Before this: Stages 8, 10 correctness checks and 11.**

**Learn:** faster token emission can still mean more retries or worse task success.

**Read:** [model card: thinking controls and sampling policies](https://huggingface.co/Qwen/Qwen3.8-27B#api-usage); your verified support matrix.

**Do next:**

1. Prepare 20 checked tasks: ten simple and ten harder, with fixed answer rules.
2. Compare thinking disabled with one supported effort/output policy and verify the settings took effect.
3. Record actual reasoning/output lengths, first useful final answer, completion, task success, truncations and failures.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A10

**core · G1 · A02/A07/A08**

Test two supported reasoning/output-budget settings using the same checkable task set. Capture actual output/reasoning lengths, task success, time to final useful answer and resource use.

**Save and check:** Completed-task comparison; verify the settings took effect and report quality/time tradeoffs, not only decode speed.

##### Practical steps — L2-D

**Consequence:** avoids optimizing raw token speed while making task completion worse.

- [ ] **L2-D01** Build a reusable 20-prompt sanity set: ten simple tasks and ten harder tasks, each with an answer check or a written scoring rubric. Include a few relevant infrastructure questions with independently checked answers.
- [ ] **L2-D02** Compare thinking disabled with one explicit supported reasoning effort. Keep the token budget adequate for the task and log truncation; save full generation settings.
- [ ] **L2-D03** Record first-output time, first-answer time where observable, completion time, generated tokens and success by prompt ID.
- [ ] **L2-D04** Investigate failures before calling the faster policy superior. Separate wrong answers, malformed output, exhausted budgets and server errors. Repeat ambiguous sampled cases only where the decision depends on them.
- [ ] **L2-D05** Recommend a policy for each workload class. State that a 20-case set is a regression check and learning tool, not a general model-quality ranking.

**Evidence:** paired task results and a policy choice. If model-card sampling recommendations differ between thinking modes, either test those complete policies and label the confound, or hold sampling fixed for an isolated experiment.

</details>

#### Stage 13 — Follow one slow interval into the code

**Core · G1 → CPU/P0 · Before this: Stage 9 and the scheduling/memory comparisons from Stage 10; A19 consolidation follows this trace.**

**Learn:** a timeline connects CPU dispatch, transfers, kernels and synchronization to latency.

**Read:** [Nsight Systems: a short CPU/GPU timeline capture](https://docs.nvidia.com/nsight-systems/UserGuide/); [vLLM engine metrics](https://docs.vllm.ai/en/latest/usage/metrics/); the pinned runtime source for the implicated path.

**Do next:**

1. Correlate a slow condition with GPU/CPU/memory telemetry and two explanations.
2. Capture a short Nsight Systems trace and an uninstrumented comparison to bound profiling overhead.
3. Follow the relevant scheduler/request function into native/kernel code; record a shape or synchronization constraint.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A12

**core · G1 · A03/A04**

Capture a short Nsight Systems trace of the chosen slow condition and an uninstrumented comparison. Correlate CPU dispatch, transfers, kernels and synchronization.

**Save and check:** Trace-backed bottleneck report with profiling-overhead caveat; identify one critical-path mechanism and one alternative explanation.

##### Q38-L02-A13

**core · CPU/P0 · A12**

Navigate from a relevant request/scheduler function to the Python/native/kernel implementation involved in the trace. Identify one shape or synchronization constraint.

**Save and check:** A source-version-bound call-path note; distinguish code navigation from kernel authorship.

##### Practical steps — L2-E

**Consequence:** builds the performance-analysis skill repeatedly relevant to NVIDIA work.

- [ ] **L2-E01** Capture timestamps alongside GPU memory/utilization, clocks/power where available, CPU usage, host RAM and engine running/waiting counts during one experiment.
- [ ] **L2-E02** Correlate a slow interval with telemetry. Propose two possible explanations and identify the observation that would distinguish them.
- [ ] **L2-E03** Capture a short steady-state Nsight Systems trace, or an engine/PyTorch trace if profiling access is unavailable. Focus on a handful of requests rather than the full model download/startup.
- [ ] **L2-E04** Identify GPU execution, CPU launch work, idle gaps and memory copies where visible. A trace can suggest bandwidth limits; proving them may need counters or a quantitative model.
- [ ] **L2-E05** Write one evidence-backed diagnosis and one limitation. If traces are blocked by the host, retain the error and use telemetry; mark the Nsight skill as pending rather than blocking the whole model study.

**Evidence:** annotated trace or telemetry interval and diagnosis. References: [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics/).

If Nsight access is blocked, save the error and use available engine/PyTorch traces or meaningful telemetry to continue a narrower diagnosis. The legacy G2 allowed that fallback. Required Q38-L02-A12/B04 Nsight evidence remains pending unless demonstrated; continuing or moving on with this named gap is allowed. Navigating source does not demonstrate kernel authorship.

</details>

#### Stage 14 — Separate client/network delay from engine work

**Core · CPU/G1 · Before this: Stages 8–9 and the request-path sketch.**

**Learn:** WAN requests, GPU collectives and worker state transfers are different paths.

**Read:** [vLLM engine metrics](https://docs.vllm.ai/en/latest/usage/metrics/); your Stage 6 request-path sketch and observed proxy/client settings.

**Do next:**

1. Run the same light workload near the server and from the normal remote client.
2. Separate connection setup from keep-alive requests; compare stream behavior and server timing where exposed.
3. Check buffering, parsing, reuse and timeouts; save a justified deployment implication.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Practical steps — L2-F

**Consequence:** turns your networking experience into inference-specific evidence.

- [ ] **L2-F01** Run the same light workload from a client near the server and from your normal remote client, using the same client version and request policy. Record route/proxy differences and client resources.
- [ ] **L2-F02** Measure connection setup separately from warm keep-alive requests where possible. Compare TTFT, total latency and streaming chunk arrival patterns.
- [ ] **L2-F03** Check whether proxy buffering, connection reuse, timeouts or client parsing could explain the difference. Use server-side timing to bound inference time when exposed.
- [ ] **L2-F04** Estimate request/response byte volume for the text workload. Explain why WAN request traffic is a different problem from GPU collectives and state transfer between serving workers.
- [ ] **L2-F05** Write a deployment implication: for example, use a regional client, preserve streaming, fix a timeout or move the load generator. Do not attribute the entire location difference to propagation delay.

**Evidence:** local/remote comparison and one justified network decision. Streaming does not require a fresh full RTT for every generated token. A TCP throughput test alone does not measure LLM response latency.

Legacy L2-F01–F05 remain core in the merged plan. Streaming does not require a new full RTT per token; do not attribute every location effect to propagation or substitute TCP throughput for response latency.

</details>

#### Stage 15 — Practice bounded admission, cancellation and recovery

**Core · CPU/G1 · Before this: Stages 8–10 correctness checks.**

**Learn:** a disconnect may leave backend work running; readiness and cleanup must be observed.

**Read:** your pinned engine cancellation/admission documentation, metrics and known-good launch configuration.

**Do next:**

1. In an isolated lab, check an over-limit request and a cancelled/timed-out stream, then verify cleanup and a later valid response.
2. Run a bounded burst and record queue/rejection behavior and recovery as demand falls.
3. Restart from cached files; measure readiness and first useful response; write a short symptom/check/recovery/validation runbook.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A14

**core · CPU/G1 · A01/A07**

In an isolated lab, cancel a streaming request and verify worker/request cleanup using available logs/metrics. Then confirm a later request succeeds.

**Save and check:** Cancellation/recovery report; distinguish client disconnect, backend cancellation and eventual resource release.

##### Q38-L02-A15

**core · G1 · A03/A14**

Run a bounded burst/admission test within the agreed safe limits. Observe queued/rejected requests, errors and recovery after load falls.

**Save and check:** Overload-boundary report; do not trigger uncontrolled OOM or affect a production/shared service.

##### Q38-L02-A16

**core · CPU/G1 · A02/A14**

Make the lab launch reproducible on the accessible host: environment/container manifest, resource limits, readiness and cold-start observations. Check what can be restarted safely.

**Save and check:** A scoped runbook and readiness check; do not claim Kubernetes/cluster operation from a single-host launch. Unavailable privileges are a documented gap.

##### Practical steps — L2-G

**Consequence:** tests whether the deployment is operable beyond the happy path.

- [ ] **L2-G01** Define readiness, a per-request timeout and bounded admission/concurrency. Record how a client sees model loading versus a ready service.
- [ ] **L2-G02** Send an over-limit request in the lab. Check rejection or truncation behaviour and verify that a normal request still works afterwards.
- [ ] **L2-G03** Cancel or time out one long request. Observe whether active-request/cache pressure recovers; do not assume a disconnected client automatically stops GPU work.
- [ ] **L2-G04** Restart the service with model files cached. Measure time to readiness and first useful response; separate this from the uncached startup.
- [ ] **L2-G05** Write a small runbook: symptom, first three checks, recovery action and validation request. Choose a real failure or a clearly labelled injected one.

**Evidence:** a recovery timeline and a usable runbook. **Stop:** one bounded lab drill is sufficient; no need to build HA infrastructure yet.

</details>

#### Stage 16 — Defend placement and useful-work economics

**Core · P0/CPU · Before this: Stages 9–15.**

**Learn:** replicas and TP trade capacity for communication; cost needs useful-completion denominators.

**Read:** [NCCL troubleshooting: topology and transports](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html); the economics reference below; your measurements.

**Do next:**

1. Map replicas versus TP onto recorded/proposed GPU, CPU, PCIe and NIC placement. Label analytical predictions.
2. Calculate successful requests/tokens per GPU-second and cost per output-token/task under the same target. Use a symbolic rate if no observed rate exists.
3. Write a one-page hosting recommendation with capacity, latency, quality, failure behavior, costs, limits and next experiment.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A17

**core · P0 · A05/A12/A13**

Compare replicas versus tensor parallelism for this workload. Map expected collectives, GPU/NIC/PCIe placement and shared request traffic onto a recorded or proposed topology.

**Save and check:** A source-bound topology/placement scenario with testable predictions; explicitly label it analytical if no G2/N2 run occurs.

##### Q38-L02-A18

**core · CPU · A03/A04/A05/A07/A10/A15**

Compute successful requests/tokens per GPU-second under the fixed latency/quality budget. Express monetary cost using a supplied rate or a symbolic rate if none is known.

**Save and check:** Capacity/economics worksheet including failures, warm-up and idle time; no invented rental price or financial outcome.

##### Practical steps — L2-H

**Consequence:** connects engineering work to consultancy and solution architecture.

- [ ] **L2-H01** Record actual GPU rental, storage and other material costs for the experiment window. Distinguish observed charges from quoted rates.
- [ ] **L2-H02** Calculate cost per million output tokens for the fixed benchmark, and cost per successful task for the sanity set. Include failures/retries in cost; define both denominators.
- [ ] **L2-H03** Compare the baseline and chosen operating configuration at the same quality/latency requirement. Label any different hardware, precision or input distribution explicitly.
- [ ] **L2-H04** Estimate what changes at low utilization, longer prompts and a burst of demand. State which estimates need measurement before a customer commitment.
- [ ] **L2-H05** Write a one-page recommendation: workload, deployment, measured capacity, latency, costs, failure behaviour, limits and next experiment.

**Evidence:** an engineering recommendation someone else can challenge and reproduce. A higher peak tokens/sec result is not automatically the cheapest useful service.

Monetary estimates are conditional on the stated rate and window. Record observed charges separately from quotes if paid compute is later used. Include warm-up, failed work/retries and idle resources; no new rental is needed for this guide.

</details>

#### Stage 17 — Transfer the method, then choose whether to move on

**Core · P0/CPU · Before this: Stages 8–16.**

**Learn:** reusing a harness is useful; transferring a mechanism still needs a new-model test.

**Read:** your own comparisons, support matrix, traces, runbook and recommendation; the next chosen model's primary configuration.

**Do next:**

1. Preserve workload IDs, runner, metric dictionary, manifest and decision format; link shared evidence instead of duplicating it.
2. State three next-model predictions: memory, performance and a distinctive mechanism. Give each generalization a condition or counterexample.
3. Independently explain a bottleneck, apply it to a load/state/failure/topology scenario and defend the evidence and cost assumptions.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A20

**core · P0 · A08/A10/A12–A19; selected extensions**

Independently explain a bottleneck, handle a practical scenario involving load/state/failure/topology, and defend the traces, quality results and cost assumptions.

**Save and check:** Private interview practice and a supported technical synthesis; list untested advanced paths rather than claiming their implementation.

##### Practical steps — L2-I

**Consequence:** prevents starting from zero on model 02.

- [ ] **L2-I01** Preserve the workload IDs, benchmark runner, metric dictionary, manifest format and decision template as reusable assets.
- [ ] **L2-I02** Use the shared concepts transfer reference and existing private evidence records. Mark each concept understood, measured or transferred; do not mark transfer before testing another model.
- [ ] **L2-I03** Record three predictions for model 02: one about memory, one about performance and one about a distinctive mechanism. Verify that model's actual configuration before relying on family names.
- [ ] **L2-I04** Separate observations specific to Qwen/GPU/runtime from conclusions likely to generalize. Give each generalization a condition or counterexample.
- [ ] **L2-I05** Pass G2 and stop expanding this model's core scope. Select the next model while retaining the same harness where valid.

**Evidence:** a portable experiment pack and a next-model handoff.

For the legacy L2-I02 concept register, use the transfer questions in this guide. U/M/T means conceptually explained / actually measured / tested on another model; keep observed personal outcomes in existing private records. Do not create a public personal completion ledger. G2 and Q38 gates are cross-referenced in B below.

</details>

#### Optional extensions — choose one for a concrete question

These belong to the Medium A list; they are separate from the normal stage sequence. A supported extension can supply the preselected memory/precision comparison while retaining its correctness and controlled-comparison checks. It cannot remove the required scheduling comparison. Two-GPU and GPU Kubernetes work remain valuable return options after early baselines, without becoming hidden graduation conditions.

#### Extension — Weight precision

**Extension · G1 · Before this: Stage 11 plus the relevant baseline/correctness evidence.**

**Learn:** format loading and accelerated execution are different claims.

**Read:** [vLLM recipe: prerequisites and the launch section for your GPU](https://recipes.vllm.ai/Qwen/Qwen3.8-27B); the derivative checkpoint's tensor/scaling/calibration provenance.

**Do next:**

1. Select one compatible derivative and verify its actual kernel path.
2. Compare memory/startup/performance with the same workload and fixed KV policy where possible.
3. Run paired quality checks and decide whether useful capacity/cost justifies the quality risk.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A06

**extension · G1 · A02/A05**

Compare the baseline with one supported weight-precision/checkpoint alternative, recording tensor/kernel and calibration provenance. Keep workload and output policy fixed.

**Save and check:** Weight-precision A/B report; do not conflate a different checkpoint, cache dtype and scheduler change into one causal claim.

##### Practical steps — L2-XQ

**Choose when:** memory is limiting, or a cheaper deployment is plausible.

- [ ] **L2-XQ01** Select one supported quantized derivative and inspect its base revision, format, scale scheme and retained higher-precision tensors. Record whether weights, activations or KV are quantized.
- [ ] **L2-XQ02** Verify that the chosen GPU/runtime has a supported execution path. Loading a format does not prove accelerated kernels are used.
- [ ] **L2-XQ03** Compare memory, startup, latency and throughput at the same workload and concurrency. Keep KV precision fixed for a weight-only comparison; flag any unavoidable change in GPU or base revision.
- [ ] **L2-XQ04** Run the paired sanity set and retain changed answers. Decide whether capacity/cost improves enough to justify the observed quality risk.

**Enough:** one credible precision tradeoff, not a survey of all quantizers. If both variants cannot fit on the same hardware, call it a deployment comparison, not an isolated precision result.

</details>

#### Extension — Prefix and state reuse

**Extension · G1 · Before this: Stage 11 plus candidate correctness checks.**

**Learn:** a hybrid cache must reuse the right recurrent, convolution and KV pieces.

**Read:** [hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/); pinned model-specific runtime support.

**Do next:**

1. Verify exact compatibility, including state slots/checkpoints and prefix boundaries.
2. Compare unique, cold shared and warm shared prefixes; save hits, eviction/overhead and correctness.
3. State reused versus recomputed work, or retain a verified support limitation.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A09

**extension · G1 · A08/A07**

If support is established, compare repeated-prefix versus cold/divergent requests and check state correctness. Include cache hits, eviction and memory overhead.

**Save and check:** Prefix/state reuse report with both correctness and latency conditions; an unsupported path stays deferred.

##### Practical steps — L2-XC

**Choose when:** repeated system prompts, documents or conversation prefixes are material.

- [ ] **L2-XC01** Check prefix-caching support for this exact hybrid model and engine revision, including recurrent-state behaviour. Save the relevant support note or observed limitation.
- [ ] **L2-XC02** Create unique-prefix, cold shared-prefix and warm shared-prefix cases with identical output policies. Reset or intentionally prime caches between cases.
- [ ] **L2-XC03** Compare TTFT, server cache-hit evidence and outputs. Check reuse at a different prefix boundary if the engine has block-alignment constraints.
- [ ] **L2-XC04** State what was reused and what was recomputed. If unsupported, document the gap and defer implementation work to L3-H.

**Enough:** measured reuse, or a verified support limitation. The [hybrid KV cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) shows why generic full-attention caching assumptions need care; the checked page still describes some Mamba-related support as work in progress. Model-specific runtime evidence takes priority over assuming blanket support or blanket absence.

</details>

#### Extension — Two GPUs and collectives

**Extension · G2 · Before this: working baseline, exact TP support and authorized two-GPU access.**

**Learn:** two GPUs do not guarantee fast peer access; total hardware must match for a fair replica comparison.

**Read:** [nccl-tests](https://github.com/NVIDIA/nccl-tests); [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).

**Do next:**

1. Record topology/affinity and verify peer/collective correctness and the selected transport.
2. Compare TP=1 and TP=2 only as an isolated scaling result if the same precision fits one GPU.
3. When both copies fit independently, compare TP=2 with two TP=1 replicas at equal GPU count and offered work.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Practical steps — L2-XT

**Choose when:** you have an affordable two-GPU window and want direct evidence for a measured infrastructure question. High priority for a later return; not required now.

- [ ] **L2-XT01** Record the physical/virtual topology, GPU form factor, PCIe/NVLink links, CPU affinity and peer-access availability. Two GPUs in one VM do not guarantee a fast peer path.
- [ ] **L2-XT02** Run an appropriate peer-bandwidth/latency check and a small `nccl-tests` sweep; verify correctness. Save NCCL version, topology and logs showing the transport actually selected.
- [ ] **L2-XT03** Compare TP=1 with TP=2 only if the same model/precision fits the one-GPU case. Record per-GPU memory, latency, throughput and total cost. Otherwise label it a capacity-enabling deployment comparison.
- [ ] **L2-XT04** If each GPU can independently fit the model, compare TP=2 against two TP=1 replicas at the same total GPU count and offered workload. Explain the tradeoff between splitting computation and serving independent requests.

**Enough:** explain one measured scaling result. Do not claim NCCL performance if the engine uses a different custom collective path. References: [NCCL tests](https://github.com/NVIDIA/nccl-tests), [NCCL diagnostics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).

</details>

#### Extension — GPU Kubernetes

**Extension · G1 on an existing GPU Kubernetes worker · Before this: known-good baseline and authorized cluster access.**

**Learn:** a device plugin exposes GPUs; probes distinguish startup from readiness.

**Read:** [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html).

**Do next:**

1. Map GPU resource request, plugin, Operator, driver/runtime and scheduler roles.
2. Deploy the known baseline to an actual GPU worker with pinned image/model, persistent cache, startup/readiness probes and limits.
3. Smoke-test, restart and correlate one pod/request problem with GPU/engine evidence.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Practical steps — L2-XK

**Choose when:** an existing GPU cluster is available or a selected deployment objective requires it.

- [ ] **L2-XK01** Explain the device plugin, GPU Operator, driver/container runtime and scheduler responsibilities. Map a requested GPU resource to a visible device inside the pod.
- [ ] **L2-XK02** Deploy the known working configuration with an explicit GPU request, persistent model cache, startup/readiness probes and request limits. Confirm the image and model revision match the baseline.
- [ ] **L2-XK03** Run a smoke load and restart the pod. Observe scheduling, image pull, model loading and readiness separately.
- [ ] **L2-XK04** Correlate pod/request behaviour with GPU/engine telemetry. Write one failure diagnosis such as an unavailable GPU, image mismatch or inappropriate startup probe.

**Enough:** one reproducible GPU workload and one recovery explanation. Use an actual GPU worker; a CPU-only local Kubernetes cluster does not validate GPU serving. Reference: [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html).

</details>

#### Extension — Speculation/MTP or one image fixture

**Extension · G1 · Before this: pinned compatibility, fixture checks and a baseline.**

**Learn:** draft verification and vision processing add work that may help only particular workloads.

**Read:** [vLLM recipe: prerequisites and the launch section for your GPU](https://recipes.vllm.ai/Qwen/Qwen3.8-27B); [SGLang recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B); exact runtime feature support.

**Do next:**

1. For compatible speculation, compare off/on at low and higher load; keep acceptance/verification overhead and correctness.
2. For an image fixture, record processor settings and separate preprocessing/vision/prefill contributions where observable.
3. Keep unsupported paths deferred; neither extension is needed for Intro/Medium core completion.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L02-A11

**extension · G1 · A08/A07**

If a compatible speculation path exists, compare ON/OFF at a low-load and a higher-load point. Record draft/verification overhead and acceptance where exposed.

**Save and check:** Speculation report; preserve target-model correctness and explain when the extra work wins or loses. Do not enable incompatible state rollback.

##### Q38-L01-A18

**extension · G1 · A10/A12**

Add one image fixture and separately time decode/preprocessing/vision work where instrumentation permits.

**Save and check:** A modality-specific correctness/timing report; no assumption that text-only latency predicts image-serving behavior.

</details>

</details>

### B — Evidence needed for Medium

<details markdown="1">
<summary>Check Medium evidence after the selected comparisons and service tests</summary>

- [ ] **Q38-L02-B01 — Reproducible protocol.** A01/A02: another run can use the pinned fixtures/environment/configuration and produce interpretable records. Material baseline differences are explained. Intro core evidence is present or re-demonstrated.
- [ ] **Q38-L02-B02 — Workload and memory-aware capacity.** A03/A05: arrival/concurrency and phase-sensitive curves, the hybrid memory ledger, and admission limits agree within explained uncertainty. I can predict a changed workload's limiting resource and say what measurement would check it.
- [ ] **Q38-L02-B03 — Controlled optimization with boundaries.** A04/A07/A19: two distinct, isolated mechanism comparisons satisfy the declared protocol, including quality/error checks and a changed-workload boundary. The conclusions may include no gain; the evidence must support the explanation. Neither a fixed percentage speedup nor every extension is required.
- [ ] **Q38-L02-B04 — Trace and implementation understanding.** A12/A13: I can locate the measured critical path and relevant source implementation, explain synchronization/data movement, and account for instrumentation overhead. This does not certify CUDA kernel engineering.
- [ ] **Q38-L02-B05 — Correctness and advanced-feature scope.** A07/A08/A10: the tested configs satisfy the frozen checks, reasoning-budget effects are verified, and support/untested features are explicitly separated. A06/A09/A11 are optional unless selected as a core comparison before work begins.
- [ ] **Q38-L02-B06 — Bounded reliability and deployment.** A14–A16: cancellation, burst/admission behavior, recovery and readiness are evidenced on the declared isolated setup. Analytical runbooks alone cannot substitute for a required cleanup/recovery observation.
- [ ] **Q38-L02-B07 — Infrastructure and economics defense.** A17/A18/A20: I can apply the model to a topology/parallelism scenario and defend successful-work capacity/cost under declared constraints. Proposed multi-GPU behavior is clearly distinguished from measured scaling. Explanation/application/defense is unaided for the selected gates.

- [ ] **Merged network/hosting/handoff evidence.** Legacy L2-F01–F05, L2-G01–G05, L2-H01–H05 and L2-I01–I05: retain the near/remote-client comparison, over-limit/restart observations, one-page hosting recommendation and reusable next-model handoff. These are core additions from the practical checklist, evidenced in Stages 14–17. Explain/apply/defend them under Q38-L02-A20 as applicable.

<details markdown="1">
<summary>Earlier G2 checklist — retained alongside the fuller checks</summary>

- [ ] **G2-01** G1 still passes with the final pinned configuration.
- [ ] **G2-02** Input, output and load sweeps explain the useful operating envelope.
- [ ] **G2-03** One controlled optimization attempt has a defensible adopt/reject decision and a correctness check. A positive speedup is not required.
- [ ] **G2-04** Reasoning/output length is included in the task-cost interpretation.
- [ ] **G2-05** One bottleneck diagnosis uses telemetry or traces, with uncertainty stated.
- [ ] **G2-06** I have compared the client/network path and demonstrated a bounded recovery procedure.
- [ ] **G2-07** A short hosting recommendation and reproducibility pack exist.
- [ ] **G2-08** I can name what the next model will teach or test.

The original G2 accepted one optimization attempt and a telemetry fallback. Full merged Medium retains two mechanism comparisons and Q38 trace/source-path evidence. A documented profiler gap allows continued study or moving on, without claiming that missing gate.

</details>

</details>

## level03-expert — keep for a chosen return project

Deferred by default. Choose a measured implementation/distributed problem, not the entire backlog. Common tasks plus one selected **K (kernel/runtime), H (hybrid state), or D (distributed inference/state transfer)** branch define the implementation route. Other branches remain deferred. Freeze hardware, correctness, failure and independent checks before measurement.

The legacy specialist modules also offer bounded investigations. A literature/code inspection or verified support limitation can close an analytical question under G3; it does not satisfy implementation gates requiring actual target-runtime changes and hardware evidence. Vision, orchestration or post-training investigations do not automatically establish K/H/D implementation completion.

### A — Learn, read and do later

<details markdown="1">
<summary>Open only when you choose an Expert return project</summary>

#### Expert setup — Freeze one project

**Later · CPU/G1; G2/N2 when the chosen project needs it · Before this: Medium core evidence.**

**Learn:** a meaningful implementation question needs a baseline, reference and failure test.

**Read:** your measured Medium gap and refreshed primary runtime/model sources.

**Do next:**

1. Choose K/H/D and write the expected deployment consequence and exit checks.
2. Reproduce on actual project hardware with pinned versions/topology.
3. Build independent correctness fixtures/reference and tolerances before tuning.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L03-A01

**common core · P0 · Medium core evidence**

Select K, H or D and one concrete question from a measured Medium gap. Define the implementation change, independent checks and success/failure conditions.

**Save and check:** Frozen project contract; explicitly list required branch IDs and deferred branches.

##### Q38-L03-A02

**common core · CPU/G1; G2/N2 for D · A01**

Recheck sources and reproduce the relevant baseline on the actual project hardware. Pin builds, commit hashes, topology and instrumentation.

**Save and check:** Reproduction/environment packet; a changed runtime invalidates unexamined support assumptions.

##### Q38-L03-A03

**common core · CPU · A01/A02**

Build minimal correctness fixtures and a reference/comparison implementation or trusted baseline for the selected change. Select numerical tolerances and failure tests before tuning.

**Save and check:** Executable tests with meaningful boundaries; do not reward a faster incorrect integration.

</details>

#### Branch K — One kernel/runtime critical path

**Later / selected k branch · CPU/G1 · Before this: Expert setup.**

**Learn:** operations, memory traffic, launches and synchronization determine the possible end-to-end gain.

**Read:** [Nsight Systems: a short CPU/GPU timeline capture](https://docs.nvidia.com/nsight-systems/UserGuide/); [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/); pinned operator/runtime source.

**Do next:**

1. Profile the implicated operator and its surrounding serving path.
2. Implement one bounded runtime/C++/CUDA/Triton change; split prototype, tests and benchmarks into sessions.
3. Compare two shapes plus an adverse boundary; retest request-level correctness and performance.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L03-K01

**branch K · G1 · A02/A03**

Profile one implicated kernel/operator with Nsight Compute and its surrounding execution with Nsight Systems. Check memory traffic, launches and synchronization against the serving trace.

**Save and check:** Kernel-plus-service diagnosis; device utilization or one microbenchmark is not end-to-end proof.

##### Q38-L03-K02

**branch K · CPU/G1 · K01**

Implement one bounded runtime/C++/CUDA/Triton change selected from the diagnosis. Split prototype, test and benchmark into separate work sessions.

**Save and check:** Actual versioned diff and local tests; clearly retain assistance and distinguish my contribution from copied code.

##### Q38-L03-K03

**branch K · G1 · K02/A03**

Compare original and changed implementations over at least two relevant shapes and an adverse boundary. Retest a serving workload.

**Save and check:** Repeated kernel and request-level evidence; explain compiler/runtime/hardware assumptions and any regression.

##### Practical steps — L3-P

**Trigger:** a timeline identifies an expensive operation or unexplained GPU idle period.

- [ ] **L3-P01** Form a quantitative bottleneck hypothesis: approximate operations and bytes moved for the selected shape. Include assumptions about weight reuse and cache traffic.
- [ ] **L3-P02** Use Nsight Systems to isolate the operation and its surrounding CPU/GPU work; distinguish startup/compilation from steady state.
- [ ] **L3-P03** If appropriate and permitted, use Nsight Compute on the selected kernel to inspect memory traffic, occupancy and execution efficiency. Record profiler overhead and sampling limits.
- [ ] **L3-P04** Evaluate one supported change such as CUDA graph mode, batch shape or backend selection. Confirm what kernel/backend ran instead of inferring it from a flag.
- [ ] **L3-P05** Only if necessary, construct a small PyTorch/Triton/CUDA reproducer. Check numerical agreement, benchmark properly and explain the end-to-end impact ceiling.

**Enough:** measured evidence for one bottleneck and a bounded improvement or negative result. Kernel coding is an advanced route, not a prerequisite for an inference solutions-architect portfolio. Start with the [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

</details>

#### Branch H — One hybrid-state operation

**Later / selected h branch · CPU/G1 · Before this: Expert setup.**

**Learn:** state ownership, lifetime, checkpointing and rollback are correctness mechanisms.

**Read:** [pinned checkpoint configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json); pinned runtime state allocation/update code.

**Do next:**

1. Trace every required recurrent/convolution/KV piece for a prefix branch, checkpoint/eviction or speculative rollback.
2. Implement or instrument the operation against clean recomputation/reference; test divergent continuations, boundaries and cleanup.
3. Measure storage/copy/checkpoint overhead and serving impact under length/slot pressure.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L03-H01

**branch H · CPU/G1 · A02/A03**

Trace state ownership/lifetime for one chosen operation: prefix branch, checkpoint/eviction or speculative rollback. Identify all required recurrent/convolution/KV pieces.

**Save and check:** Source-bound state-lifecycle diagram and test design; incomplete state capture is a correctness failure.

##### Q38-L03-H02

**branch H · CPU/G1 · H01**

Implement or instrument the selected state operation against an independent recomputation/reference path. Test divergent continuations, boundaries and cleanup.

**Save and check:** Versioned code and correctness results with stated tolerances; a CPU toy can guide design but target-runtime G1 evidence is still required.

##### Q38-L03-H03

**branch H · G1 · H02**

Measure the operation's storage, copy/checkpoint overhead and serving impact. Vary request lengths and slot/branch pressure safely.

**Save and check:** State/performance comparison and failure boundary; confirm reuse remains correct under the tested load.

##### Practical steps — L3-H

**Trigger:** memory scaling, prefix reuse or a state-management bug is unexplained.

- [ ] **L3-H01** Trace the runtime's DeltaNet and attention state allocation from configuration to tensor shapes. Separate conceptual state from reserved engine pools.
- [ ] **L3-H02** Estimate recurrent-state bytes from the actual layer/head dimensions and storage dtype, including convolution state. Compare with runtime tensor inspection where accessible.
- [ ] **L3-H03** Locate prefill and decode paths, identifying how recurrent state is produced, updated and retained. Explain why parallel prefill need not execute like sequential decode.
- [ ] **L3-H04** Test one state boundary: prefix reuse, cancellation, preemption or checkpointing. Compare output/correctness against a clean run.
- [ ] **L3-H05** Produce an annotated explanation or minimal bug reproducer with exact versions and conditions.

**Enough:** one state-management question resolved or isolated to a specific implementation limit. No need to recreate the model from scratch.

</details>

#### Branch D — One distributed inference/state-transfer question

**Later / selected d branch · G2 for single-host; N2 for cross-host claims · Before this: Expert setup and verified hardware/support.**

**Learn:** collectives, worker state transfers and client traffic need separate evidence.

**Read:** [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html); [Dynamo routing](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing); [Dynamo disaggregation](https://docs.nvidia.com/dynamo/cli/disaggregated-serving/overview).

**Do next:**

1. Verify exact TP/disaggregation support and complete hybrid-state handoff; record ranks/GPU/NIC/transports.
2. Configure or implement one controlled experiment with transfer/collective size, time, synchronization and overlap.
3. Change one condition and test an isolated worker/rank/transfer failure with recovery and corruption checks.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L03-D01

**branch D · G2/N2 · A02/A03**

Verify exact model/runtime support for the selected tensor-parallel or disaggregated path and its complete hybrid-state handoff. Map ranks, GPUs, NICs and transports.

**Save and check:** Compatibility/topology packet; no assumption that generic KV-transfer support covers recurrent/convolution state. A missing capability blocks that project.

##### Q38-L03-D02

**branch D · G2/N2 · D01**

Implement or configure one controlled multi-GPU/distributed experiment. Measure collectives or transfer sizes/time, synchronization and overlap separately from request-path traffic.

**Save and check:** Actual setup, trace and raw comparison; record whether observations are single-host or cross-host. Do not fabricate fabric measurements.

##### Q38-L03-D03

**branch D · N2 for cross-host project; G2 for single-host project · D02**

Change one topology/transfer/load condition and test one isolated rank/worker or transfer failure with a safe recovery path.

**Save and check:** Scaling/transfer/failure evidence; distinguish availability failures from silent output/state corruption. A cross-host claim requires N2 evidence.

##### Practical steps — L3-N

**Trigger:** a distributed model deployment, a selected networking investigation or a real communication bottleneck requires it.

- [ ] **L3-N01** Confirm actual RDMA-capable NIC access, fabric connectivity, driver support and provider permissions before renting multiple nodes. If unavailable, design the lab and mark execution pending.
- [ ] **L3-N02** Inventory GPU–NIC/CPU topology, NIC link rate, NUMA placement, routing and interface names. Distinguish NVLink inside a system from InfiniBand/RoCE between systems.
- [ ] **L3-N03** Run fabric-appropriate latency/bandwidth checks and an NCCL message-size sweep with correctness validation. Do not equate TCP `iperf` bandwidth with RDMA or collective performance.
- [ ] **L3-N04** Confirm selected transport from logs and counters. Check for unexpected socket fallback, wrong interfaces, poor affinity or disabled GPU peer access before changing tuning variables.
- [ ] **L3-N05** Relate collective performance to inference step time using a distributed workload or trace. Treat all-reduce, all-gather and all-to-all as different traffic patterns.
- [ ] **L3-N06** On an isolated fabric you control, investigate a single congestion/placement issue; for RoCE, consider the actual ECN/PFC policy and counters. Do not modify shared network policy to complete a study task.

**Enough:** explain one end-to-end distributed inference result with topology, transport and communication evidence. Qwen's dense backbone can teach tensor-parallel collectives; expert routing/all-to-all belongs with an appropriate MoE model. References: [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html), [NCCL tests](https://github.com/NVIDIA/nccl-tests).

##### Practical steps — L3-D

**Trigger:** long prefills disrupt decode, or a multi-worker architecture decision needs evidence.

- [ ] **L3-D01** Verify the serving framework/connector supports the exact hybrid model and transfers all required state. Full-attention-only examples do not prove DeltaNet support.
- [ ] **L3-D02** Estimate attention-cache and recurrent-state transfer bytes per handoff. Divide by achievable, measured link bandwidth for an ideal transfer lower bound, then account for protocol and scheduling costs.
- [ ] **L3-D03** Compare colocated and disaggregated serving at equal total hardware and a fixed workload. Include routing, serialization, transfer and queue time.
- [ ] **L3-D04** Test the relevant failure/retry path and observe whether a failed transfer causes recomputation or request failure.
- [ ] **L3-D05** Recommend when disaggregation earns back its communication and operational cost. If support is absent, select another model for the experiment and preserve the architectural question.

**Enough:** a bounded break-even analysis with measured inputs, or a verified compatibility limitation. This can become an NVIDIA Dynamo/NIXL study later; product support must be checked when selected.

</details>

#### Expert closure — Retest, package and defend

**Later / selected project common core · actual project hardware → P0 · Before this: K03 or H03 or D03.**

**Learn:** operator-level evidence must connect back to correct useful requests.

**Read:** your implementation diff, independent tests, traces and workload comparisons.

**Do next:**

1. Retest correctness, latency, quality, capacity/resource use and one changed-condition boundary.
2. Package versioned code/config/results/environment references and reproducible instructions.
3. Independently explain, apply and defend the selected claim; consider a focused upstream contribution without claiming submission or acceptance.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Q38-L03-A04

**common core · project hardware · K03 or H03 or D03**

Retest correctness, latency, completed-task quality, capacity and resource use for the selected implementation. Compare at the frozen workload and at a boundary.

**Save and check:** End-to-end implementation report including negative results and uncertainty; connect mechanisms to service impact and cost.

##### Q38-L03-A05

**common core · P0 · A04**

Package a reproducible technical artifact with code/config/raw-result/environment references, then independently explain, apply and defend it. Review whether a focused upstream contribution is worthwhile.

**Save and check:** Reproduction instructions and private evidence defense; proposing a contribution does not authorize submitting it or claim acceptance.

</details>

#### Later investigation — Speculation and MTP

**Later · G1 · Before this: verified compatibility plus baseline.**

**Learn:** acceptance is only one part of drafting benefit.

**Read:** [vLLM recipe: prerequisites and the launch section for your GPU](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

**Do next:**

1. Verify draft/runtime/precision compatibility and correctness guarantees.
2. Compare off/on at concurrency 1 and a loaded point; record acceptance, verification, memory and useful-request latency.
3. Choose a setting or plain-decoding fallback.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Practical steps — L3-S

**Trigger:** low-concurrency decode latency dominates and a compatible drafting path exists.

- [ ] **L3-S01** Verify draft weights, runtime support, precision compatibility and correctness guarantees for the selected method. Distinguish a trained MTP head from enabled runtime speculation.
- [ ] **L3-S02** Compare drafting off with one small speculative-token setting under a fixed workload and reasoning policy.
- [ ] **L3-S03** Record accepted/drafted tokens, extra memory, total latency and throughput at concurrency 1 and at a loaded operating point.
- [ ] **L3-S04** Explain whether draft generation, verification or rejected tokens dominate. Higher acceptance alone does not prove a speedup.
- [ ] **L3-S05** Keep or disable speculation using measured benefit and output checks. Preserve a plain decoding fallback.

**Enough:** establish the workload conditions where speculation helps or hurts. Use the hardware-specific [Qwen vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) as a starting point; do not transplant its reported acceptance or throughput.

</details>

#### Later investigation — Orchestration and reliability

**Later · actual GPU cluster / multi-worker setup · Before this: known workload and authorized cluster.**

**Learn:** deployment events affect useful capacity through load, readiness and draining.

**Read:** [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)

**Do next:**

1. Define a multi-worker deployment with GPU placement, storage, secrets, probes and graceful shutdown.
2. Correlate user-impact metrics and test bounded overload plus one rollout/failure/scale event.
3. Evaluate a placement/autoscaling choice, including cold starts and draining.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Practical steps — L3-O

**Trigger:** a customer, role exercise or sustained service requires Kubernetes and multiple workers.

- [ ] **L3-O01** Turn the known workload into a deployment spec covering GPU placement, persistent storage, secrets, startup/readiness and graceful shutdown.
- [ ] **L3-O02** Establish correlated engine, GPU and node metrics; choose alerts tied to user impact rather than GPU utilization alone.
- [ ] **L3-O03** Test overload and bounded admission with realistic arrival patterns. Measure useful completions, latency and rejected demand.
- [ ] **L3-O04** Test one rollout, worker failure or scale event. Account for cold starts, loading time and request draining.
- [ ] **L3-O05** Evaluate one orchestration choice, such as replica placement or an autoscaling signal, with measured data.

**Enough:** one operational design and failure exercise a reviewer can inspect. Do not claim production reliability from a short lab. Reference: [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html).

</details>

#### Later investigation — Vision and long context

**Later · G1 or verified larger target hardware · Before this: working baseline and application need.**

**Learn:** allocated context and successful long-context answers are different evidence.

**Read:** [model card: Model Overview](https://huggingface.co/Qwen/Qwen3.8-27B#model-overview); exact processor/runtime compatibility

**Do next:**

1. Record controlled image resolution, processor settings, visual tokens and encoder execution.
2. Separate upload/preprocessing/vision/prefill where observable; extend text length safely.
3. Check retrieval/reasoning quality at each length and declare a supported application envelope.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Practical steps — L3-V

**Trigger:** a real workload needs images or contexts beyond your baseline.

- [ ] **L3-V01** Add one controlled image workload; record source resolution, preprocessing, visual-token accounting and encoder execution.
- [ ] **L3-V02** Measure upload/preprocessing/encoder/prefill contributions where observable. Record processor settings so the same image gives a comparable workload.
- [ ] **L3-V03** Extend text context in safe steps with a bounded output budget. Measure cache/state pressure, latency and task quality at each step.
- [ ] **L3-V04** Compare meaningful retrieval/reasoning tasks at those lengths. A successful allocation or advertised context length does not demonstrate effective long-context quality.

**Enough:** a supported context/modality envelope for the chosen application. Context extension beyond the native limit is a separate compatibility and quality experiment.

</details>

#### Later investigation — Post-training, judged by serving consequences

**Later · appropriate serving hardware; separate training project if chosen · Before this: measured residual failure and explicit project/resource scope.**

**Learn:** custom behavior has training cost and changes serving memory, speed and output length.

**Read:** primary model/runtime adapter documentation for the selected route

**Do next:**

1. Define target behavior, held-out checks and a serving budget; test prompting/constrained output/reasoning first.
2. Only if justified, compare base, adapter-served and merged deployment where supported.
3. Account for RL rollout/update compatibility if selected and accept only held-out gains justified by training/serving costs.

<details markdown="1">
<summary>Small steps, task IDs and evidence checks</summary>

Work through one small step at a time. The IDs below are references to the same work, so reuse the same artifact wherever it satisfies several checks.

##### Practical steps — L3-T

**Trigger:** the model misses a specific task requirement, or behaviour creates avoidable serving cost.

- [ ] **L3-T01** Define the target behaviour, held-out evaluation set and acceptable serving budget before considering SFT, preference tuning or reinforcement learning.
- [ ] **L3-T02** Establish whether prompting, constrained output or reasoning policy already solves the problem. Record the residual failure that justifies training.
- [ ] **L3-T03** If an adapter experiment is justified, compare base, adapter-served and merged deployment where supported. Measure memory, throughput, correctness and output-length changes.
- [ ] **L3-T04** For RL-style work, estimate rollout-generation cost and model-update/serving compatibility. Treat the training pipeline as its own project with a separate budget.
- [ ] **L3-T05** Accept customization only if held-out task gains justify training and serving costs. Track task success and completion time together.

**Enough:** one validated behaviour change and its serving impact. Full RLHF/RL training is not a gate for moving beyond this first inference model.

</details>

</details>

### B — Evidence needed for the selected Expert project

<details markdown="1">
<summary>Implementation gates and earlier G3 investigation checks</summary>

- [ ] **Q38-L03-B01 — Frozen scope and authentic baseline.** A01/A02: Medium evidence is present, the selected branch and its exact hardware/correctness/failure requirements are fixed, and the baseline is reproduced on appropriate hardware.
- [ ] **Q38-L03-B02 — Real implementation and correctness.** A03 plus K01–K03 **or** H01–H03 **or** D01–D03: actual target-runtime code/configuration and independent correctness checks demonstrate the selected branch. Reading papers, changing unrelated flags or presenting a surrogate alone is insufficient.
- [ ] **Q38-L03-B03 — Mechanism and hardware evidence.** The selected branch's traces/raw results show the relevant kernel/state/communication mechanism, overhead, units and conditions. GPU work requires GPU artifacts; cross-host/fabric claims require cross-host/fabric artifacts.
- [ ] **Q38-L03-B04 — End-to-end boundary and reproducibility.** A04/A05: the implementation comparison includes the agreed correctness/failure tests, changed-condition boundary, uncertainty and reproduction packet. A negative performance finding is acceptable when the implementation question is answered and limits are defended.
- [ ] **Q38-L03-B05 — Independent technical defense.** A05: explain the concept, apply it to a new serving/infra scenario and defend code/measurements without hints for the selected claim. Identify deferred branches. This is bounded subject evidence, not a statement of professional Expert/Senior readiness or production-scale operating experience.

##### Earlier G3 — a bounded specialist investigation

- [ ] **G3-01** The original question and its deployment consequence are written down.
- [ ] **G3-02** The explanation is supported by a trace, code inspection, controlled measurement or a verified support limitation.
- [ ] **G3-03** Another engineer can reproduce the key observation from the saved setup.
- [ ] **G3-04** The conclusion states scope, uncertainty and what would trigger a revisit.

Close the module and move on. Completing every Level 03 module is not a meaningful gate.

G3 may close an analytical/support question; the Q38 implementation gates require authentic implementation/hardware evidence. Closing one chosen module never requires exhausting all later modules.

</details>

## Reference shelf — open only what the current stage needs

<details markdown="1">
<summary>Stage 2 reference — worked memory estimates and model dimensions</summary>

The examples here are direct explanations for reference. Try your own prediction before using them as a check; reading them alone is not independent evidence.

| Property | Verified reference value | Inference question |
|---|---|---|
| Checkpoint | `Qwen/Qwen3.8-27B` | Which exact revision and precision am I serving? |
| Model class | Dense language backbone with a vision encoder | Which components are actually loaded and exercised? |
| Backbone | 64 layers: 48 Gated DeltaNet and 16 gated full-attention layers | Which state grows with history? |
| Full attention | 24 query heads; 4 KV heads; head dimension 256 | What is the KV payload per token? |
| Native context | 262,144 tokens | What smaller limit does my workload actually need? |
| Reasoning | Thinking defaults on; effort controls are provided | What do I count as output and time to answer? |
| Drafting | MTP is part of the released architecture | Is it supported, activated and beneficial in my exact runtime? |

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B), [checkpoint configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).

The current [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) includes hardware-specific precision and serving examples. Use the recipe matching your GPU, then pin the successful version. A recipe's headline memory or throughput number is not a prediction for a different GPU or workload. Validate optional feature combinations independently.

### Memory calculations you should be able to reproduce

Approximate BF16 weights, using the nominal parameter count:

`27 × 10^9 parameters × 2 bytes ≈ 54 GB ≈ 50.3 GiB`

Full-attention KV payload, BF16, one sequence, tensor parallelism 1:

`16 layers × 2 [K,V] × 4 KV heads × 256 dimensions × 2 bytes = 65,536 bytes/token = 64 KiB/token`

| Resident sequence length | Attention KV payload per sequence |
|---|---:|
| 2,048 tokens | 128 MiB |
| 8,192 tokens | 512 MiB |
| 32,768 tokens | 2 GiB |

Four independent sequences at 8,192 resident tokens therefore have 2 GiB of attention KV payload. Count input plus generated history at the observation point, not just prompt tokens.

These are calculated payloads, not measurements of allocated GPU memory. Add recurrent and convolution state, any retained vision/MTP weights, workspaces, CUDA graph allocations, allocator overhead and serving-engine cache layout. Actual tensor inventories beat nominal parameter labels. Padding, sharing, replication and cache reservation can change the observed footprint.

The pinned config declares `linear_num_key_heads: 16`, `linear_num_value_heads: 48`, key/value head dimensions 128, convolution kernel dimension 4 and `mamba_ssm_dtype: float32`. The actual recurrent/convolution tensors, effective storage dtype and multiple state slots/checkpoints must still be checked in the selected runtime. [Pinned configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json).

</details>

<details markdown="1">
<summary>Stages 4–16 reference — experiments, metrics and economics</summary>

These are suggested workloads, not performance targets. All lengths below are token counts after the model's actual template/tokenizer. Keep input plus output within the configured limit. Reduce the largest point if hardware cannot sustain it.

| Experiment | Change | Hold fixed | Primary evidence |
|---|---|---|---|
| **E00 — baseline** | Repeat one warm workload | About 512 input / 128 output, C=1 | Raw timings, actual lengths, memory, errors |
| **E01 — input length** | 512 / 2,048 / 8,192 input | 128 output, C=1, cache policy | TTFT versus length; attention/state pressure |
| **E02 — output length** | 128 / 512 / 1,024 output | 2,048 input, C=1 | Completion time and decode behaviour |
| **E03 — load** | C=1 / 2 / 4 / 8 / optionally 16; then one arrival-rate check | 2,048 input / 256 output | Throughput–latency envelope and queueing |
| **E04 — tuning** | One scheduler/memory knob, baseline plus two settings | GPU, precision, policy and mixed workload | Tradeoff between throughput, TTFT and ongoing decode |
| **E05 — reasoning** | Thinking disabled versus one supported effort | Same 20 tasks and scoring rules | Success, generated length, time and cost |
| **E06 — network path** | Near-server versus remote client | Client implementation and light workload | Connection/stream timing plus server evidence |
| **E07 — operations** | Over-limit request, cancellation, restart | Known good deployment | Recovery/readiness timeline |
| **E08 — economics** | Compare measured operating choices | Defined quality/SLO requirement | Cost per successful task and useful capacity |

For E04, one useful synthetic mix is 80% short prompts and 20% long prompts, using the input sizes already established in E01 and a common output target. Save the exact schedule/seed. It is a stress experiment until you have evidence that the mix represents a customer workload.

### Keep the experiments small but credible

- Start with 10 requests per exploratory point to catch gross effects; this is enough for a smoke test, not a robust p95 claim.
- For a decision you intend to publish or use commercially, a practical starting point is at least 100 completed requests per condition over multiple independent runs. Report sample count and spread; more data may be needed for unstable workloads. Do not claim a precise p99 from this small study.
- Warm the engine consistently. Separate cold cache from warm cache and cold startup from steady state.
- Use the same prompt IDs, actual token lengths, output policy, reasoning policy and arrival process for paired conditions. Alternate comparison order when drift or shared-host noise matters.
- For synthetic throughput tests, fixed generation length can be useful. For quality tests, preserve normal stopping and count incomplete answers as failures.
- Save errors, timeouts, cancellations and early stopping. Excluding slow failed requests makes a bad service look faster.
- Make sure the load generator is not the bottleneck. Remote networking is a deliberate variable in E06, not an accidental limiter in every experiment.
- End sweeps once they answer the question. Do not run all possible combinations of context, precision, concurrency and engine settings.

### Metric definitions to keep with every result

| Metric | Definition or rule |
|---|---|
| Client TTFT | Send → first non-empty generated output observed by the client; distinguish it from HTTP headers or empty metadata chunks |
| Time to first final-answer content | Send → first final-answer content; meaningful separately when reasoning is emitted or hidden |
| End-to-end latency | Send → complete response, including queueing and transport from that client's perspective |
| TPOT estimate | `(completion time − first-output time) / (output tokens − 1)`, only when output count and timing describe the same token stream and N>1 |
| Inter-token latency | Consecutive token intervals if actually observable; network chunk intervals are a different measurement |
| Aggregate output throughput | Sum of generated output tokens across completed requests divided by the declared observation window; state treatment of partial/failed requests |
| Request goodput | Requests meeting the declared latency/error criteria per second; optionally also require a defined quality check |
| SLO pass rate | Fraction meeting the specified SLO, with offered/admitted/completed denominator explicitly named |
| KV payload | Logical bytes of cached keys/values; separate from recurrent state and preallocated GPU cache memory |
| Cost per successful task | Total cost for the declared workload window, including failed work/retries, divided by successful tasks; report undefined if none succeed |

For hybrid inference, `nvidia-smi` memory may remain almost flat as live sequence lengths change because the server has already allocated its cache pools. Use the engine's cache occupancy and allocation reporting to interpret the payload calculation. See [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/).

### Two useful economics checks

For a steady synthetic workload, if hourly compute cost is `H` and sustained aggregate output throughput is `R`:

`compute cost / million output tokens = H × 1,000,000 / (3,600 × R)`

This is a workload-specific compute estimate. It does not include idle capacity, storage, operations or different input costs unless incorporated into H and the measurement window.

For task economics:

`cost / successful task = total relevant cost during the evaluation window / count of successful tasks`

Measure reasoning and retry effects in the second formula. Do not use token throughput alone to conclude that the faster engine configuration produces the cheaper answer.

For small comparison claims, keep **at least three independent repetitions per condition**, record sample counts/spread and predeclare latency/quality budgets. Ten requests are a smoke point. At least 100 completed requests per condition across runs is the earlier practical starting suggestion for a public/commercial decision, not proof of a robust p95/p99 or a universal minimum. Count failed/partial work explicitly; do not silently discard slow failures.

</details>

<details markdown="1">
<summary>Evidence reference — one report, configuration manifest and experiment template</summary>

Keep one canonical report under [experiments/](experiments/); day notes link to it. Record prediction before measurement, changed/controlled variables, cold/warm state, repetitions, actual observations, uncertainty and failure boundary. Include checkpoint/tokenizer/runtime/container/driver/toolkit versions, hardware/topology, weight/KV/recurrent-state dtypes, fixtures/seeds, load and lengths, settings/limits, commands, correctness, latency/throughput/errors and quality. Keep credentials out of logs/manifests. Use numerical tolerances where appropriate; stochastic output equality is not a universal correctness rule.

Suggested future repository locations; these are organizational recommendations, not files already created by this checklist:

| Location | Contents |
|---|---|
| `models/01-qwen3.8-27b/README.md` | Model navigation and links; personal gate outcomes remain private |
| `models/01-qwen3.8-27b/configs/` | Pinned manifests and launch configurations |
| `models/01-qwen3.8-27b/results/` | Raw results and concise comparisons |
| `models/01-qwen3.8-27b/profiles/` | Selected traces or links to large captures |
| Existing shared private study notes | Shared explanations linking to experiments; no new public root needed |
| Existing private shared benchmark location, or ignored model-local experiment files | Reusable workloads and runner |
| `days/<existing-day-folder>/` | Session notes linking to the model and concept records |

Do not copy model weights or massive traces into Git. Keep small summaries and durable references to the larger evidence. One concise report can hold several experiments; a separate document per checkbox is unnecessary.

### Configuration manifest template

```yaml
# Fill with observed values; null means not yet recorded.
run_id: null
timestamp_utc: null

# Pin model and tokenizer identity independently if they differ.
model_id: Qwen/Qwen3.8-27B
model_revision: null
tokenizer_revision: null
chat_template_hash: null

# Record the execution environment, not just the GPU marketing family.
provider_region: null
gpu_sku: null
gpu_count: 1
usable_vram_gib: null
cpu_and_numa: null
host_ram_gib: null
topology_evidence: null
driver_version: null
cuda_runtime_version: null
container_digest: null
engine_version_or_commit: null
benchmark_version_or_commit: null

# Record effective settings, including defaults that affect comparisons.
weight_format: null
activation_dtype: null
kv_dtype: null
recurrent_state_dtype: null
tensor_parallel_size: 1
max_model_len: null # 16384 is a possible initial lab limit, not an observed/default value
max_num_seqs: null
max_num_batched_tokens: null
gpu_memory_utilization: null
prefix_cache_policy: null
cuda_graph_policy: null
speculation_policy: disabled
offload_policy: disabled
loaded_vision_and_mtp_components: null
launch_command: null

# Capture workload and client behaviour for causal comparisons.
workload_id_and_hash: null
thinking_policy: null
sampling_parameters: null
output_stopping_policy: null
client_location: null
arrival_process: null
requested_load: null
actual_completed_load: null
warmup_and_cache_state: null
sample_count: null

# Distinguish measured facts from assumptions in the result notes.
predicted_bottleneck: null
observed_bottleneck: null
cost_basis: null
evidence_paths: []
limitations: []
```

### One experiment record

```markdown
<!-- This is a template, not a completed experiment. -->
# E__ — question

- Task IDs:
- Workload and SLO:
- Prediction and confidence:
- Baseline run ID:
- Changed variable:
- Controlled variables:
- Known confounders:
- Results and sample count:
- Failures/cancellations/truncations:
- Quality check:
- Interpretation and alternative explanations:
- Adopt/reject/defer decision:
- What transfers to another model:
- Evidence paths:
- Return trigger:
```

Configurations, code, raw results and large traces remain local/ignored until explicitly selected for publication. Only authorized public targets belong in public report links. One report can cover several comparisons; a new document per checkbox is unnecessary.

</details>

<details markdown="1">
<summary>Transfer reference — shared concepts and questions for the next model</summary>

The earlier U/M/T shorthand distinguished explained, measured and transferred evidence. Transfer means predicting/diagnosing another model, including finding that an earlier assumption fails. This reference lists questions; actual personal outcomes stay in existing private records.

| Concept | Evidence to reuse / later question |
| --- | --- |
| Weights versus runtime memory | L1-B; do parameter labels predict loaded bytes? |
| KV cache, GQA and recurrent state | L1-B/G; L3-H if needed |
| Prefill versus decode | E01/E02 |
| HBM bandwidth versus compute | L2-E; L3-P for proof |
| Continuous batching and queueing | E03 |
| Chunked prefill and scheduling budgets | E04 if selected |
| Reasoning length and useful task cost | E05/E08 |
| Quantization and execution kernels | L1-B; L2-XQ when selected |
| Prefix reuse and cache correctness | L2-XC when supported |
| CUDA graphs and host launch overhead | L2-E; L3-P |
| Network path versus GPU service time | E06 |
| GPU/CPU/NIC topology and NUMA | L1-G; L2-XT |
| Tensor parallelism versus replicas | L2-XT |
| NCCL/collectives and RDMA transport | L2-XT; L3-N |
| Speculative decoding acceptance and cost | L3-S |
| Prefill/decode state handoff | L3-D |
| Readiness, overload and recovery | E07; L3-O |
| Capacity planning and goodput | E03/E08 |

Reuse rule: for a later model, rerun a compact baseline and one shared diagnostic experiment. Spend new study time on its distinctive mechanism. Do not replay demonstrated Qwen tasks unless material versions, assumptions or behavior changed.

</details>

<details markdown="1">
<summary>Later systems reference — GPU tools and depth priorities</summary>

Keep measurement/diagnosis, GPU memory/topology, communication, Linux/container operation, workload-based recommendations and Python runner engineering connected to actual inference questions. C++/CUDA kernel contribution and platform/network architecture are separate depth choices. GPU Kubernetes, real two-GPU comparisons and multi-node RDMA remain useful selected return projects; they are not prerequisites for the first baseline.

| Layer | Tools/products to recognize | Evidence to seek |
|---|---|---|
| Execution | CUDA and TensorRT-LLM; vLLM as the first engine | Which runtime/backend executes the selected model and why |
| Packaging/deployment | Containers and NVIDIA NIM where applicable | Supported model/hardware profile, configuration and operational behaviour |
| Distributed serving | NVIDIA Dynamo and state-transfer components such as NIXL | Routing/state movement costs and exact compatibility |
| GPU orchestration | GPU Operator and device plugins | GPU scheduling, visibility, readiness and recovery |
| Observability | Nsight Systems/Compute; DCGM where available | Correlated measurements that explain user-visible latency or failures |
| Communication | NCCL, NVLink, InfiniBand and RoCE | Actual transport/topology and measured communication behaviour |

This is an orientation map, not a claim that Qwen3.8-27B is supported in every listed product. Check the exact support matrix before selecting a lab. For engine orientation, see [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html).

</details>

<details markdown="1">
<summary>Parking lot — how to keep a question without making it a prerequisite</summary>

| Question | Why it matters | Evidence needed | Return trigger | Status |
|---|---|---|---|---|
| Example: why does long-context latency jump? | May change the supported workload | Trace plus cache/preemption metrics | A repeated discontinuity in E01 | Unstarted |
| Example: would TP=2 beat two replicas? | GPU/network architecture choice | Same-hardware comparison | Two suitable GPUs available | Unstarted |
| Example: will a smaller dtype help? | Capacity and rental cost | Paired runtime/quality comparison | Weight or cache memory limits E03 | Unstarted |

Return when the answer could change a deployment decision, explain a repeated anomaly, meet a concrete role requirement or support a useful contribution. Curiosity is welcome; record it without making it a hidden prerequisite.

**Final stopping rule:** after all merged Medium B checks, you should be able to say: “For this workload, on this hardware and software revision, I can reproduce the service, explain the main bottleneck, defend the operating configuration and state its limits.” That is enough for model 01.

The statement is bounded to the recorded workload/hardware/software. You can move on with missing evidence named; the missing gate remains undemonstrated.

</details>

<details markdown="1">
<summary>Primary source directory</summary>

Read only the sections required by the current experiment. These links were checked on 13 September 2026; implementations and support combinations can change.

| Reference | Use |
|---|---|
| [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) | Model identity, behaviour, generation controls |
| [Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | Architecture dimensions and configured dtypes |
| [vLLM Qwen3.8-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | Hardware-specific launch and compatibility starting points |
| [vLLM benchmark serve CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/) | Benchmark invocation and supported options |
| [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) | Scheduler/memory tradeoffs |
| [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) | Engine-side observability |
| [vLLM hybrid KV cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) | Hybrid allocation and support caveats |
| [NVIDIA Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | Focused CPU/GPU timeline capture |
| [NVIDIA NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) | Topology, GPU/NIC and fabric diagnosis |
| [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) | Collective correctness and performance tests |
| [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) | GPU Kubernetes infrastructure |
| [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html) | NVIDIA inference-runtime orientation |

Numerical memory examples are calculations from published dimensions; workload sizes, timeboxes, tasks and progression criteria are proposed learning-design choices, not benchmark claims from those sources.

Also use the pinned checkpoint config, model-specific SGLang recipe, Nsight Compute, Dynamo routing/disaggregation and the selected runtime source linked in the stages above. Moving `main`/`latest` pages are planning references. Resolve exact revisions again at execution time.

</details>

## When to move on

| Situation | Next action |
| --- | --- |
| All Intro B checks are demonstrated | Start Medium Stage 8, or choose another Intro model comparison. Reuse evidence. |
| Reading is done but the target GPU run is absent | State the exact pending support/hardware task. You can change subjects; hands-on Intro remains pending. |
| All Medium B checks plus merged network/hosting/handoff evidence are demonstrated | This is enough depth for the bounded workload. Move to the next chosen model; leave Expert deferred. |
| A core measurement or profiler privilege is missing | Name the gap and one follow-up. Continue a narrower study or move by choice, without treating the gap as demonstrated. |
| A measured issue justifies Expert | Select one K/H/D implementation project, or a named analytical investigation, and freeze its distinct exit checks. |
| A new GPU/runtime/precision/workload invalidates an assumption | Refresh primary support, reproduce the relevant baseline and rerun the affected comparison only. |

Independent explanation, scenario application and evidence defense are part of the selected gates. In a coached session, hear the complete first response before hints and ask one focused question at a time. End a study day with that same explain → apply → defend perspective, using only work actually done. Keep observed evaluations/assistance in canonical private records; this technical plan does not create a score store, claim hiring readiness or override formal assessments.

**Start now:** write your serving question and record the accessible GPU. That is the first session; Stage 2 follows from what you find.
