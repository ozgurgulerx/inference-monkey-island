md # List 01 — Qwen3.8-27B study tasks and completion gates

Sources checked: **2026-09-13**. This is an uncompleted study plan. My current route is **level01-intro → level02-medium → the next chosen model**. **level03-expert is a return path**, not a requirement before studying GLM-4.7-Flash.

I have access to an NVIDIA GPU. Its model, VRAM, topology and permissions have not yet been recorded. Start with that inventory; do not assume BF16 deployment fits. No task here authorizes renting compute, modifying a shared service or spending money.

## Planning references and assumptions

The [official model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json) supplies the hybrid layer schedule, attention geometry, linear-state geometry and declared dtypes. The model's implementation identifiers still use `qwen3_5`; verify the selected artifacts rather than deriving runtime support from the marketing name. Treat the supplied diagram as an orientation aid, then compare it with configuration and code. Do not automatically rewrite my existing notes.

The [official model card](https://huggingface.co/Qwen/Qwen3.8-27B) describes thinking/reasoning controls and preserved history. Runtime behavior must be checked: an accepted API field is not evidence that a setting actually changed. The [vLLM Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) and [SGLang Qwen recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) are model-specific starting points. Their hardware/version examples are not guarantees for my GPU.

Use [vLLM benchmark documentation](https://docs.vllm.ai/en/latest/cli/bench/serve/) to select metrics and request generation. Trace interpretation can use [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/) and, later, [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/). General [Dynamo routing](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing) and [disaggregation documentation](https://docs.nvidia.com/dynamo/cli/disaggregated-serving/overview) distinguish worker selection from state transfer. Verify Qwen's complete hybrid-state compatibility before applying either design.

All latest-documentation links are planning references. Pin the actual checkpoint, runtime/container and configuration used in experiments. No throughput number, context limit or precision choice is guaranteed by this plan.

## How I will use the lists

**A lists** specify work and deliverables. **B lists** specify when the evidence is sufficient for that level's stated scope. Every box starts unchecked. A completed reading task is not a completed GPU experiment.

Most rows are 15–45 minutes of active work; split a larger row into setup, run, analysis and explanation sessions. GPU runtime, downloads and debugging are additional unknowns. Each row ends at its stated artifact, not at an open-ended investigation of the whole topic.

Hardware labels: **P0** = source/paper work; **CPU** = executable host-side work; **G1** = one suitable NVIDIA GPU; **G2** = multiple GPUs on one host; **N2** = at least two suitable hosts with recorded network topology. A CPU toy, provider response, borrowed trace or diagram cannot satisfy a G1/G2/N2 measurement gate.

**Core** tasks are required by that level's B gates. **Extension** tasks can be deferred. Expert branch tasks become required only for the branch selected before the project begins. Preserve stable task IDs when revising this plan.

Before each coached task I give my prediction, attempt and first explanation. AI feedback follows the complete attempt. For an experimental conclusion, I need a mechanism and a boundary, not just a faster number. A negative result is useful evidence when the comparison is sound.

### Common experiment packet

Keep one report per experiment under [experiments/](experiments/), and link to it from the day notes. Record target checkpoint/revision; runtime and container versions; driver/toolkit; hardware/topology; weight, KV and recurrent-state precision; workloads and input/output lengths; arrival rate/concurrency; limits; cold/warm state; prediction; commands/configuration; raw measurements and repetitions; correctness checks; latency/throughput/errors; uncertainty; and where the result stops holding. Keep secrets out of commands, logs and manifests.

For small comparisons, use at least three independent repetitions per condition and report sample counts and spread. Freeze workload fixtures and an explicit latency/quality budget before interpreting a change. A small sample's p95 is descriptive, not a production guarantee. Raw files and executable code remain local/ignored until specifically selected for publication; public report links must have authorized public targets.

## level01-intro

Scope: identify the actual architecture, account for its important state, correctly serve the target model and measure a small text-only baseline. Vision, advanced caching, speculation, custom kernels and multi-GPU deployment can wait.

### A — Microtasks to study and do

| Task / scope / hardware / dependencies | Microtask | Deliverable and check |
| --- | --- | --- |
| [ ] Q38-L01-A01 · core · P0 · none | State my first serving question and freeze a text-only Intro scope. Pick a small input/output envelope, task-correctness rule and latency target I can test. | A short experiment contract; separate chosen targets from facts about the model. |
| [ ] Q38-L01-A02 · core · CPU/G1 · A01 | Inventory the accessible GPU, VRAM, driver/toolkit, host RAM, CPU, device topology and permissions. Identify whether this is an isolated lab or shared resource. | Environment manifest without credentials; list unavailable profiling or deployment privileges rather than guessing. |
| [ ] Q38-L01-A03 · core · CPU · A02 | Pin the exact model repository/revision and a supported runtime version. Inspect the model config, tokenizer/processor and runtime recipe. | Artifact/version manifest with primary links; identify model and implementation names, supported dtypes and unresolved support questions. |
| [ ] Q38-L01-A04 · core · P0 · A03 | Annotate one repeated block in the model map: token mixing, normalization, FFN, residuals, attention positional encoding, and where vision/MTP are omitted. Count layer types from config. | My own annotated execution path; every numerical geometry claim is config-bound, with any diagram disagreement noted. |
| [ ] Q38-L01-A05 · core · CPU · A03 | Predict weight storage from actual tensor count and selected bytes per element. Convert GB/GiB and separate checkpoint size from GPU runtime allocation. | A small executable memory worksheet; quantify omitted metadata, scales and runtime overhead. Do not turn the 27B label into a full-capacity prediction. |
| [ ] Q38-L01-A06 · core · CPU · A04/A05 | Derive full-attention KV growth from the actual KV heads, head dimension, layer count, cache dtype and live tokens. Predict two context/concurrency cases. | Unit-checked KV worksheet; distinguish token-linear KV from fixed-size recurrent state and from allocator reservation. |
| [ ] Q38-L01-A07 · core · P0/CPU · A03/A06 | Locate linear-attention recurrent and convolution state shapes and runtime slot/checkpoint handling. Account for state dtype and possible multiple slots per request. | A source-bound state ledger; explicitly leave runtime-dependent unknowns unresolved until startup logs or code confirm them. |
| [ ] Q38-L01-A08 · core · CPU · A01/A03 | Build a tiny text fixture set: short request, longer prompt, controlled output budget, multi-turn chat, stop/cancellation boundary, and one checkable task. | Saved fixtures and validation rules; use deterministic input fixtures, not a universal promise of byte-identical stochastic outputs. |
| [ ] Q38-L01-A09 · core · G1 · A02–A08 | Start the target model with a conservative supported configuration that fits the inventory. Choose a documented precision if BF16 is unsuitable and label it. Record startup memory and admitted limits. | Launch/configuration and startup report; no unapproved shared-service changes. A fit failure remains evidence to diagnose, not a successful deployment. |
| [ ] Q38-L01-A10 · core · G1 · A08/A09 | Check the actual chat template, assistant response, stop handling, reasoning/final output separation and multi-turn behavior. | Correctness results for the saved fixtures; show that the integration is operating as intended before timing it. |
| [ ] Q38-L01-A11 · core · G1 · A09/A10 | Measure startup, first-request and warmed low-concurrency request behavior separately. | Cold/warm timing table with a defined warm-up; avoid reporting initialization cost as steady-state decode performance. |
| [ ] Q38-L01-A12 · core · CPU/G1 · A10/A11 | Capture client-side time to first token, output count, total latency and an appropriate steady-output metric. Record errors and units. | Baseline table plus measurement definitions; account for streaming granularity and distinguish client timings from engine metrics. |
| [ ] Q38-L01-A13 · core · G1 · A12 | Change only input length for a small prefill-focused comparison while holding output policy and load constant. Predict the result first. | Repeated input-length comparison; explain what is actually isolated and what still confounds the result. |
| [ ] Q38-L01-A14 · core · G1 · A12 | Change output budget for a decode-focused comparison while holding prompt fixtures and load constant. Record actual generated length. | Output-length comparison; explain why output limits, early stops and reasoning tokens affect completed-task latency. |
| [ ] Q38-L01-A15 · core · G1 · A06/A07/A12 | Try a modest concurrency sweep within the safe envelope. Collect throughput, request latency, memory and errors with the same request mix. | A small capacity curve; identify the first observed tradeoff or limit without deliberately exhausting a shared GPU. |
| [ ] Q38-L01-A16 · core · CPU/G1 · A13–A15 | Capture host/GPU utilization and memory alongside one slow condition. Sketch client → queue → CPU work → GPU execution. State a bottleneck hypothesis and its alternative. | A correlated observation plus a path diagram; do not call utilization alone proof of compute, memory or network saturation. |
| [ ] Q38-L01-A17 · core · P0 · A04–A16 | Explain the architecture/state story, apply it to a changed prompt/load scenario, and defend the measurements and limits without AI hints. | My unaided explanation/application/defense in private practice, plus a supported technical takeaway in my own notes when I choose to write it. |
| [ ] Q38-L01-A18 · extension · G1 · A10/A12 | Add one image fixture and separately time decode/preprocessing/vision work where instrumentation permits. | A modality-specific correctness/timing report; no assumption that text-only latency predicts image-serving behavior. |

### B — Evidence needed to call Intro complete

- [ ] **Q38-L01-B01 — Source-bound scope and architecture.** A01–A04: the manifest identifies exact target/runtime/hardware, and I can trace the hybrid block and its omissions. Unresolved support needed for the baseline is resolved or explicitly blocks completion.
- [ ] **Q38-L01-B02 — Memory/state reasoning.** A05–A07/A09: my worksheet distinguishes weights, attention KV, recurrent/convolution state and runtime reservation; units and assumptions are checked against artifacts and startup observations. I can explain why fitting weights does not establish serving capacity.
- [ ] **Q38-L01-B03 — Correct target-model run.** A08–A10: the actual Qwen checkpoint runs on the recorded GPU/configuration and the declared text fixture checks pass. A surrogate, provider-only API or copied report does not satisfy this gate.
- [ ] **Q38-L01-B04 — Measured baseline and phase sensitivity.** A11–A14: raw records and the stated repetition protocol support cold/warm, input-length and output-length comparisons with understandable metric definitions. I can explain confounders; no fixed tok/s threshold is required.
- [ ] **Q38-L01-B05 — Capacity and plausible diagnosis.** A15/A16: the bounded load comparison records latency, throughput, memory and errors, and I can defend a bottleneck hypothesis plus a test that could disprove it. A diagnosis can remain provisional if the next measurement is specified.
- [ ] **Q38-L01-B06 — Interview perspective.** A17: independently explain the concept, apply it to a changed serving scenario and defend the actual evidence without claiming vision, multi-GPU or kernel expertise. A18 is optional and not an Intro graduation condition.

## level02-medium

Scope: reproducible, workload-aware serving experiments; measured scheduling and memory/precision choices; trace-backed diagnosis; correctness, reliability and economics. Stay on the existing GPU unless the selected question needs more hardware.

### A — Microtasks to study and do

| Task / scope / hardware / dependencies | Microtask | Deliverable and check |
| --- | --- | --- |
| [ ] Q38-L02-A01 · core · CPU · Intro core evidence | Freeze the Medium workload/SLO/quality envelope and the selected comparison branches. Build a reusable runner with recorded fixtures, settings, response validation and structured result output. | Executable runner and protocol; explicitly select scheduling plus one distinct memory/precision/admission mechanism for controlled testing. |
| [ ] Q38-L02-A02 · core · G1 · A01 | Reproduce the Intro baseline with the runner, then repeat a condition. Check sample counts, spread, warm-up and timing overhead. | Reproduction report; diagnose a material difference before comparing a tuning change. |
| [ ] Q38-L02-A03 · core · G1 · A02 | Sweep a small arrival-rate/request-mix grid separately from a concurrency-limit sweep. Include long-prefill/short-decode and short-prefill/long-decode cases. | Latency/throughput/error curves; distinguish generated demand, admitted requests, completed requests and queueing. |
| [ ] Q38-L02-A04 · core · G1 · A03 | Test one documented scheduling lever, such as chunked prefill or batched-token limits, in a pinned runtime. Hold other knobs and workload constant. | A/B scheduling report; account for throughput and decode/request latency instead of reporting only the faster case. |
| [ ] Q38-L02-A05 · core · CPU/G1 · A02/A03 | Refine the state/KV/overhead ledger using runtime pool/slot observations. Predict a supported admission, sequence-limit or memory-pool change and measure its capacity effect. | Predicted-versus-observed capacity table; reconcile live use and reserved allocation, including hybrid-state slots when applicable. |
| [ ] Q38-L02-A06 · extension · G1 · A02/A05 | Compare the baseline with one supported weight-precision/checkpoint alternative, recording tensor/kernel and calibration provenance. Keep workload and output policy fixed. | Weight-precision A/B report; do not conflate a different checkpoint, cache dtype and scheduler change into one causal claim. |
| [ ] Q38-L02-A07 · core · CPU/G1 · A04/A05; A06 if chosen | Run the declared quality/correctness fixtures on every selected candidate configuration. Check stop behavior, output truncation, failed requests and task success. | Quality/error comparison; a faster configuration outside the frozen quality budget is rejected for that workload. |
| [ ] Q38-L02-A08 · core · P0/CPU · A05 | Build a support matrix for prefix reuse, recurrent-state precision, branching/checkpointing, MTP/speculation and full hybrid-state transfer on the pinned runtime/model. | Primary-source/code-backed support matrix; separate supported, tested, unknown and unavailable. An accepted flag is insufficient. |
| [ ] Q38-L02-A09 · extension · G1 · A08/A07 | If support is established, compare repeated-prefix versus cold/divergent requests and check state correctness. Include cache hits, eviction and memory overhead. | Prefix/state reuse report with both correctness and latency conditions; an unsupported path stays deferred. |
| [ ] Q38-L02-A10 · core · G1 · A02/A07/A08 | Test two supported reasoning/output-budget settings using the same checkable task set. Capture actual output/reasoning lengths, task success, time to final useful answer and resource use. | Completed-task comparison; verify the settings took effect and report quality/time tradeoffs, not only decode speed. |
| [ ] Q38-L02-A11 · extension · G1 · A08/A07 | If a compatible speculation path exists, compare ON/OFF at a low-load and a higher-load point. Record draft/verification overhead and acceptance where exposed. | Speculation report; preserve target-model correctness and explain when the extra work wins or loses. Do not enable incompatible state rollback. |
| [ ] Q38-L02-A12 · core · G1 · A03/A04 | Capture a short Nsight Systems trace of the chosen slow condition and an uninstrumented comparison. Correlate CPU dispatch, transfers, kernels and synchronization. | Trace-backed bottleneck report with profiling-overhead caveat; identify one critical-path mechanism and one alternative explanation. |
| [ ] Q38-L02-A13 · core · CPU/P0 · A12 | Navigate from a relevant request/scheduler function to the Python/native/kernel implementation involved in the trace. Identify one shape or synchronization constraint. | A source-version-bound call-path note; distinguish code navigation from kernel authorship. |
| [ ] Q38-L02-A14 · core · CPU/G1 · A01/A07 | In an isolated lab, cancel a streaming request and verify worker/request cleanup using available logs/metrics. Then confirm a later request succeeds. | Cancellation/recovery report; distinguish client disconnect, backend cancellation and eventual resource release. |
| [ ] Q38-L02-A15 · core · G1 · A03/A14 | Run a bounded burst/admission test within the agreed safe limits. Observe queued/rejected requests, errors and recovery after load falls. | Overload-boundary report; do not trigger uncontrolled OOM or affect a production/shared service. |
| [ ] Q38-L02-A16 · core · CPU/G1 · A02/A14 | Make the lab launch reproducible on the accessible host: environment/container manifest, resource limits, readiness and cold-start observations. Check what can be restarted safely. | A scoped runbook and readiness check; do not claim Kubernetes/cluster operation from a single-host launch. Unavailable privileges are a documented gap. |
| [ ] Q38-L02-A17 · core · P0 · A05/A12/A13 | Compare replicas versus tensor parallelism for this workload. Map expected collectives, GPU/NIC/PCIe placement and shared request traffic onto a recorded or proposed topology. | A source-bound topology/placement scenario with testable predictions; explicitly label it analytical if no G2/N2 run occurs. |
| [ ] Q38-L02-A18 · core · CPU · A03/A04/A05/A07/A10/A15 | Compute successful requests/tokens per GPU-second under the fixed latency/quality budget. Express monetary cost using a supplied rate or a symbolic rate if none is known. | Capacity/economics worksheet including failures, warm-up and idle time; no invented rental price or financial outcome. |
| [ ] Q38-L02-A19 · core · CPU/G1 · A04/A05/A07/A12 | Consolidate two controlled changes with different mechanisms: scheduling plus memory/admission or supported precision. Retest the original and a changed workload. | One comparative report linking existing experiments; explain causality and a no-gain/regression boundary. Separate individual changes before testing a combined configuration. |
| [ ] Q38-L02-A20 · core · P0 · A08/A10/A12–A19; selected extensions | Independently explain a bottleneck, handle a practical scenario involving load/state/failure/topology, and defend the traces, quality results and cost assumptions. | Private interview practice and a supported technical synthesis; list untested advanced paths rather than claiming their implementation. |

### B — Evidence needed to call Medium complete

- [ ] **Q38-L02-B01 — Reproducible protocol.** A01/A02: another run can use the pinned fixtures/environment/configuration and produce interpretable records. Material baseline differences are explained. Intro core evidence is present or re-demonstrated.
- [ ] **Q38-L02-B02 — Workload and memory-aware capacity.** A03/A05: arrival/concurrency and phase-sensitive curves, the hybrid memory ledger, and admission limits agree within explained uncertainty. I can predict a changed workload's limiting resource and say what measurement would check it.
- [ ] **Q38-L02-B03 — Controlled optimization with boundaries.** A04/A07/A19: two distinct, isolated mechanism comparisons satisfy the declared protocol, including quality/error checks and a changed-workload boundary. The conclusions may include no gain; the evidence must support the explanation. Neither a fixed percentage speedup nor every extension is required.
- [ ] **Q38-L02-B04 — Trace and implementation understanding.** A12/A13: I can locate the measured critical path and relevant source implementation, explain synchronization/data movement, and account for instrumentation overhead. This does not certify CUDA kernel engineering.
- [ ] **Q38-L02-B05 — Correctness and advanced-feature scope.** A07/A08/A10: the tested configs satisfy the frozen checks, reasoning-budget effects are verified, and support/untested features are explicitly separated. A06/A09/A11 are optional unless selected as a core comparison before work begins.
- [ ] **Q38-L02-B06 — Bounded reliability and deployment.** A14–A16: cancellation, burst/admission behavior, recovery and readiness are evidenced on the declared isolated setup. Analytical runbooks alone cannot substitute for a required cleanup/recovery observation.
- [ ] **Q38-L02-B07 — Infrastructure and economics defense.** A17/A18/A20: I can apply the model to a topology/parallelism scenario and defend successful-work capacity/cost under declared constraints. Proposed multi-GPU behavior is clearly distinguished from measured scaling. Explanation/application/defense is unaided for the selected gates.

## level03-expert — Deferred return path

Scope: return only for a chosen implementation problem. Completing one selected branch means **the selected Expert project is demonstrated**, not that every Qwen feature, CUDA topic, fabric or production operating condition has been mastered. Medium core evidence is a prerequisite.

Choose one branch before work starts: **K — kernel/runtime critical path**, **H — hybrid-state implementation**, or **D — distributed inference/state transfer**. The common core and the chosen branch's tasks become required; other branches stay deferred. Freeze the project's exact correctness, hardware and failure gates before measurements.

### A — Microtasks to study and do later

| Task / scope / hardware / dependencies | Microtask | Deliverable and check |
| --- | --- | --- |
| [ ] Q38-L03-A01 · common core · P0 · Medium core evidence | Select K, H or D and one concrete question from a measured Medium gap. Define the implementation change, independent checks and success/failure conditions. | Frozen project contract; explicitly list required branch IDs and deferred branches. |
| [ ] Q38-L03-A02 · common core · CPU/G1; G2/N2 for D · A01 | Recheck sources and reproduce the relevant baseline on the actual project hardware. Pin builds, commit hashes, topology and instrumentation. | Reproduction/environment packet; a changed runtime invalidates unexamined support assumptions. |
| [ ] Q38-L03-A03 · common core · CPU · A01/A02 | Build minimal correctness fixtures and a reference/comparison implementation or trusted baseline for the selected change. Select numerical tolerances and failure tests before tuning. | Executable tests with meaningful boundaries; do not reward a faster incorrect integration. |
| [ ] Q38-L03-K01 · branch K · G1 · A02/A03 | Profile one implicated kernel/operator with Nsight Compute and its surrounding execution with Nsight Systems. Check memory traffic, launches and synchronization against the serving trace. | Kernel-plus-service diagnosis; device utilization or one microbenchmark is not end-to-end proof. |
| [ ] Q38-L03-K02 · branch K · CPU/G1 · K01 | Implement one bounded runtime/C++/CUDA/Triton change selected from the diagnosis. Split prototype, test and benchmark into separate work sessions. | Actual versioned diff and local tests; clearly retain assistance and distinguish my contribution from copied code. |
| [ ] Q38-L03-K03 · branch K · G1 · K02/A03 | Compare original and changed implementations over at least two relevant shapes and an adverse boundary. Retest a serving workload. | Repeated kernel and request-level evidence; explain compiler/runtime/hardware assumptions and any regression. |
| [ ] Q38-L03-H01 · branch H · CPU/G1 · A02/A03 | Trace state ownership/lifetime for one chosen operation: prefix branch, checkpoint/eviction or speculative rollback. Identify all required recurrent/convolution/KV pieces. | Source-bound state-lifecycle diagram and test design; incomplete state capture is a correctness failure. |
| [ ] Q38-L03-H02 · branch H · CPU/G1 · H01 | Implement or instrument the selected state operation against an independent recomputation/reference path. Test divergent continuations, boundaries and cleanup. | Versioned code and correctness results with stated tolerances; a CPU toy can guide design but target-runtime G1 evidence is still required. |
| [ ] Q38-L03-H03 · branch H · G1 · H02 | Measure the operation's storage, copy/checkpoint overhead and serving impact. Vary request lengths and slot/branch pressure safely. | State/performance comparison and failure boundary; confirm reuse remains correct under the tested load. |
| [ ] Q38-L03-D01 · branch D · G2/N2 · A02/A03 | Verify exact model/runtime support for the selected tensor-parallel or disaggregated path and its complete hybrid-state handoff. Map ranks, GPUs, NICs and transports. | Compatibility/topology packet; no assumption that generic KV-transfer support covers recurrent/convolution state. A missing capability blocks that project. |
| [ ] Q38-L03-D02 · branch D · G2/N2 · D01 | Implement or configure one controlled multi-GPU/distributed experiment. Measure collectives or transfer sizes/time, synchronization and overlap separately from request-path traffic. | Actual setup, trace and raw comparison; record whether observations are single-host or cross-host. Do not fabricate fabric measurements. |
| [ ] Q38-L03-D03 · branch D · N2 for cross-host project; G2 for single-host project · D02 | Change one topology/transfer/load condition and test one isolated rank/worker or transfer failure with a safe recovery path. | Scaling/transfer/failure evidence; distinguish availability failures from silent output/state corruption. A cross-host claim requires N2 evidence. |
| [ ] Q38-L03-A04 · common core · project hardware · K03 or H03 or D03 | Retest correctness, latency, completed-task quality, capacity and resource use for the selected implementation. Compare at the frozen workload and at a boundary. | End-to-end implementation report including negative results and uncertainty; connect mechanisms to service impact and cost. |
| [ ] Q38-L03-A05 · common core · P0 · A04 | Package a reproducible technical artifact with code/config/raw-result/environment references, then independently explain, apply and defend it. Review whether a focused upstream contribution is worthwhile. | Reproduction instructions and private evidence defense; proposing a contribution does not authorize submitting it or claim acceptance. |

### B — Evidence needed to call the selected Expert project complete

- [ ] **Q38-L03-B01 — Frozen scope and authentic baseline.** A01/A02: Medium evidence is present, the selected branch and its exact hardware/correctness/failure requirements are fixed, and the baseline is reproduced on appropriate hardware.
- [ ] **Q38-L03-B02 — Real implementation and correctness.** A03 plus K01–K03 **or** H01–H03 **or** D01–D03: actual target-runtime code/configuration and independent correctness checks demonstrate the selected branch. Reading papers, changing unrelated flags or presenting a surrogate alone is insufficient.
- [ ] **Q38-L03-B03 — Mechanism and hardware evidence.** The selected branch's traces/raw results show the relevant kernel/state/communication mechanism, overhead, units and conditions. GPU work requires GPU artifacts; cross-host/fabric claims require cross-host/fabric artifacts.
- [ ] **Q38-L03-B04 — End-to-end boundary and reproducibility.** A04/A05: the implementation comparison includes the agreed correctness/failure tests, changed-condition boundary, uncertainty and reproduction packet. A negative performance finding is acceptable when the implementation question is answered and limits are defended.
- [ ] **Q38-L03-B05 — Independent technical defense.** A05: explain the concept, apply it to a new serving/infra scenario and defend code/measurements without hints for the selected claim. Identify deferred branches. This is bounded subject evidence, not a statement of professional Expert/Senior readiness or production-scale operating experience.

## When it is enough and when to move on

| Decision | Evidence/condition | Next action |
| --- | --- | --- |
| Intro → Medium | All Intro B gates demonstrated for the recorded text/GPU scope. | Start Medium A01; reuse existing artifacts rather than repeating reading or creating duplicate reports. |
| Intro → another model | Intro gates demonstrated and I want a fast architecture survey. | Move by choice, with Qwen Medium left open. The next listed model is GLM-4.7-Flash unless I choose otherwise. |
| Concepts covered, GPU evidence missing | Source/analytical tasks done but the target run or measurement gates are absent/blocked. | Name the exact pending task and hardware/support gap. I can move on; do not label Intro hands-on complete. |
| Medium → another model | All Medium B gates demonstrated in the frozen bounded scope. | This is the default stopping depth for the current fast pass. Move to the next chosen model; leave Expert deferred. |
| Medium gaps, but I choose to move | A specific B gate lacks evidence or access. | Keep the gap explicit in private canonical practice context and nominate one follow-up. Moving on is a study choice, not a completion award. |
| Medium → Expert | I choose a specific implementation or distributed-inference question worth the extra work. | Select branch K/H/D and freeze A01. No requirement to exhaust every advanced feature first. |
| Return to Expert later | A repeated bottleneck, correctness issue, infrastructure constraint or chosen implementation objective warrants it. | Recheck sources/runtime/hardware, reproduce the relevant baseline, then resume the selected branch. |

Private evaluator notes, assistance conditions, observed gate outcomes and numeric assessments belong in the existing canonical private assessment system, not this plan or another score store. Do not infer completion from boxes checked without artifacts. Formal canonical exit requirements still apply independently. Publication of this technical plan is separate from completion or readiness assessment.

## My first microtask

Start with **Q38-L01-A01**, then **A02**: write the serving question and record the GPU/environment. Next, predict the memory fit before inspecting the runtime's observed allocation. Keep the prediction and my first explanation; AI hints come after the attempt.
