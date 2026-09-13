# list01 — Qwen3.8-27B: practical inference study

**Project:** inference-a-new-hope / inference-planet-arrakis  
**Owner:** Ozgur Guler  
**Version:** 1.0 · 13 September 2026  
**Sources checked:** 13 September 2026  
**Status:** A study and experiment plan. No deployment or benchmark results are claimed here.

## 1. The outcome and the stopping point

Build a reproducible Qwen3.8-27B deployment, understand its main resource demands, measure its behaviour under load, and make a defensible hosting decision. Every concept should support a prediction, a measurement, a diagnosis, or a deployment choice.

**Recommended route: Level 01 → Level 02 core → model 02, GLM-4.7-Flash.** Level 03 is a return backlog. Level 02 extensions are optional. Neither must be completed before changing models.

**Checklist size:** 28 Level 01 actions; 45 Level 02 core actions; 16 optional Level 02 actions; 40 deferred Level 03 actions. Gate checkboxes are additional. Only the first two core sets are the normal path for this model.

This is an effective path because each new model can test an existing concept and introduce a new mechanism. Completing this file demonstrates a bounded piece of applied competence; expertise comes from repeating the work across architectures and operating conditions.

### Start with these five actions

1. Read the model facts below and complete L1-A.
2. Estimate memory in L1-B.
3. Choose the smallest practical deployment configuration in L1-C.
4. Host it in L1-D. Do this before exploring architecture internals further.
5. Capture the first useful baseline in L1-E through L1-G.

**Preparation stop:** spend roughly 45–90 minutes on L1-A through L1-C, then start hosting. If a concept remains unclear, record a prediction with a confidence level and let an experiment help.

### What each level means

| Level | Observable capability | Working timebox, excluding downloads and access problems | Enough when… |
|---|---|---|---|
| **level01-intro** | Run and measure one model; explain its basic memory and request path | Roughly 3–5 focused hours | Gate G1 passes |
| **level02-medium** | Diagnose bottlenecks, evaluate a change, establish a useful capacity envelope | Roughly 8–16 focused hours for the core; split into small sessions | Gate G2 passes |
| **level03-depth** | Investigate one specialist mechanism with code, traces or distributed experiments | One selected module at a time | That module's research question is answered |

Timeboxes are planning estimates, not deadlines or reasons to skip evidence. Avoid turning a small lab into a production platform build.

### Progression decisions

| Decision | Rule |
|---|---|
| Level 01 → Level 02 | Pass G1. No architecture derivations or CUDA programming required. |
| Level 02 → next model | Pass G2, save the evidence pack, and name the concept the next model will test. This is the default. |
| Level 01 → next model early | Only for a concrete comparison or a documented hardware/runtime blocker. Save the baseline and mark G2 incomplete; this is an intentional detour. |
| Level 02 → Level 03 | A measured bottleneck, target job, customer requirement or contribution opportunity needs that depth. Select one module and define its stopping condition before starting. |
| Level 03 → next model | Close the selected question, including an inconclusive result with limits. Do not finish the whole backlog. |
| Return to Qwen later | A new engine, GPU, quantization, workload or model comparison invalidates a recorded assumption. Rerun the affected experiment only. |

## 2. How to use the checkboxes

- A checkbox represents a small action with observable evidence. Reading without answering the associated question is not completion.
- Most preparation actions should take 5–20 minutes. GPU experiments may take 20–60 minutes plus setup. Split an action further if it regularly exceeds an hour.
- For experiments, write **prediction → controlled change → observation → explanation → decision**.
- Use `N/A — reason` for an unsupported feature and `BLOCKED — requirement` for unavailable hardware. Neither means the capability was demonstrated.
- Keep unknowns in the return backlog. Do not silently convert a plausible explanation into a measurement.
- A rejected optimization can be a successful experiment. You are learning to make decisions, not collecting positive speedup claims.

Each task has a stable ID. Add the evidence filename or result ID beside it when ticking it off. Reuse your existing day folders for chronology and a model folder for consolidated conclusions. Shared concepts should link to experiments rather than duplicate their raw results.

## 3. Model facts to ground the lab

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

## 4. level01-intro — host and establish a baseline

All groups in this section are core. Finish these before chasing optimizations.

### L1-A — Identify the computation and state

**Consequence:** prevents applying the wrong memory or parallelism model.

- [ ] **L1-A01** Save the checkpoint identifier and resolve its actual revision SHA. Record tokenizer/processor revisions if different.
- [ ] **L1-A02** Locate `layer_types`, `num_key_value_heads`, `head_dim` and `dtype` in the configuration. Annotate their serving implications in one sentence each.
- [ ] **L1-A03** Explain the distinction between dense FFNs and hybrid attention. State why expert parallelism is not the natural scaling mechanism for this backbone.
- [ ] **L1-A04** List the state kept by full attention versus DeltaNet. State that a fixed recurrent-state shape does not make the entire hybrid model constant-memory with context.

**Evidence:** a short model fact sheet. **Stop:** you can explain the request's state without deriving DeltaNet.

### L1-B — Estimate memory before choosing hardware

**Consequence:** avoids choosing a GPU from parameter count alone.

- [ ] **L1-B01** Reproduce the BF16 weight estimate above. Distinguish decimal GB from binary GiB.
- [ ] **L1-B02** Reproduce the KV calculation and the four-sequence example. Explain why KV heads, rather than query heads, enter that formula.
- [ ] **L1-B03** Sketch a memory budget with separate rows for weights, attention cache, recurrent state, graph/workspace allocations and safety headroom. Mark unknown rows explicitly.
- [ ] **L1-B04** Compare idealized 16-, 8- and 4-bit weight payloads. Explain why scale metadata, mixed precision and kernel support prevent assuming exact halving of runtime memory or latency.

**Evidence:** one small memory table, including assumptions. **Stop:** choose a feasible first configuration; do not demand an exact allocator prediction.

### L1-C — Specify the first lab

**Consequence:** makes failures and measurements interpretable.

- [ ] **L1-C01** Choose one NVIDIA GPU, one serving engine and one precision. Preferred reference: a sufficiently provisioned 80 GB-class GPU with BF16. If cost/capacity dictates otherwise, use a documented compatible quantized checkpoint and label it as the baseline.
- [ ] **L1-C02** Start with text-only requests, TP=1, concurrency=1, maximum context around 16,384, no speculative decoding and no weight offload. Explicitly record cache behaviour and whether the runtime still loads vision/MTP components.
- [ ] **L1-C03** Define a rental spend cap, storage allowance and shutdown point. Include download/startup/idle time in the estimate; stop the GPU while doing lengthy reading.
- [ ] **L1-C04** Write a configuration manifest: GPU/SKU/VRAM, CPU/RAM, driver, CUDA runtime, container digest, engine version, checkpoint revision, dtypes, launch arguments and client location. Unknown hardware details remain unknown.

**Evidence:** a completed configuration manifest. **Stop:** no need to select the cheapest GPU in the market or install Kubernetes.

### L1-D — Bring up a reproducible endpoint

**Consequence:** establishes the minimum operational skill behind inference work.

- [ ] **L1-D01** Confirm GPU visibility inside the runtime and record available VRAM. Check that driver/runtime compatibility and the model backend match the selected recipe.
- [ ] **L1-D02** Download/load the checkpoint using a persistent cache. Separate weight acquisition, model loading, compilation and readiness times where logs permit.
- [ ] **L1-D03** Start the server, retain startup logs and send one short text request. Check the returned model identity, output, finish reason and token usage.
- [ ] **L1-D04** Set thinking explicitly for the request and verify the observed response. Use a private endpoint or authenticated access path; record timeouts and the streaming setting.

**Evidence:** launch command, successful response and startup log. **Stop:** you have a working endpoint and know how to start it again.

### L1-E — Establish measurement semantics

**Consequence:** avoids confusing network chunks, reasoning and GPU decode speed.

- [ ] **L1-E01** Record send time, first non-empty generated-content arrival and completion time using a monotonic clock. Do not count a role-only or empty initial chunk as the first generated token.
- [ ] **L1-E02** Distinguish first generated output from first final-answer content when thinking is enabled. If the API hides reasoning tokens, record the observability limit.
- [ ] **L1-E03** Record actual input/output tokens and distinguish per-request output rate from aggregate output throughput. Label chunk gaps separately from true inter-token latency if chunks contain multiple tokens.
- [ ] **L1-E04** Compare the benchmark tool's metric definitions with your own. Use the engine's documented benchmark utility where practical; save its version and arguments.

**Evidence:** a metric dictionary and one raw per-request record. See [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/).

### L1-F — Run E00: the warm baseline

**Consequence:** creates the reference for every later claim.

- [ ] **L1-F01** Warm the engine with several requests. Keep cold-start timing separate and record whether compilation or graph capture continues during measurement.
- [ ] **L1-F02** Run a small fixed workload, approximately 512 input tokens and 128 output tokens, concurrency 1. Save actual lengths; a maximum-output limit does not force that length.
- [ ] **L1-F03** Repeat ten requests for a first smoke baseline. Record successful completions, TTFT, completion latency, output rate, GPU memory and obvious errors. Do not present ten requests as a reliable tail-latency study.
- [ ] **L1-F04** Rerun one batch to detect a major warm-up or environmental effect. If results differ, annotate the likely cause and uncertainty.

**Evidence:** E00 raw results and a compact summary. **Stop:** the baseline is credible enough to compare experiments, even if it is slow.

### L1-G — Map infrastructure and close Level 01

**Consequence:** starts the networking thread immediately without forcing distributed inference.

- [ ] **L1-G01** Sketch the path: client, network/proxy, API process, tokenizer/scheduler, GPU, response stream. Mark which parts you can observe.
- [ ] **L1-G02** Capture `nvidia-smi topo -m` if available. Identify PCIe/NVLink indications and CPU/NUMA affinity; on a one-GPU instance, explicitly record that GPU collective behaviour was not tested.
- [ ] **L1-G03** Compare estimated weights/cache with startup allocations. Explain at least the major categories of the gap; a preallocated cache pool will not track live tokens linearly in `nvidia-smi`.
- [ ] **L1-G04** Stop and restart the service from the saved configuration, then complete the same smoke request. Save a brief unresolved-questions list.

**Evidence:** request-path sketch, topology capture and restart note.

### Gate G1 — Enough to leave Level 01

- [ ] **G1-01** I can load the pinned configuration and get a valid response again.
- [ ] **G1-02** I can explain the weight estimate, growing attention cache and additional state.
- [ ] **G1-03** I have a warm baseline with explicit token counts and measurement definitions.
- [ ] **G1-04** I can distinguish client/network delay, queueing, prefill and decode conceptually.
- [ ] **G1-05** I have one performance prediction to test in Level 02.

**If all five pass, proceed.** Do not wait to understand all model code or every launch flag.

## 5. level02-medium — explain, improve and operate

Complete L2-A through L2-I for the core. The extension groups in section 6 are separate and optional. The experiment catalog in section 8 gives concrete workload shapes.

### L2-A — E01/E02: separate prompt and generation costs

**Consequence:** reveals which workloads strain prefill or decode.

- [ ] **L2-A01** Predict the effect of increasing prompt length at fixed concurrency and fixed generated length.
- [ ] **L2-A02** Run E01 at roughly 512, 2,048 and 8,192 input tokens with 128 generated tokens. Control prefix reuse; record actual lengths.
- [ ] **L2-A03** Run E02 at roughly 128, 512 and 1,024 generated tokens with a fixed 2,048-token input. Use a controlled generation-length facility for synthetic throughput tests if supported; keep natural stopping for quality tests.
- [ ] **L2-A04** Plot or tabulate TTFT, completion time and output rate against input/output length. Separate engine measurements from client observations.
- [ ] **L2-A05** Explain the observed direction using compute, memory traffic and state access. Treat “prefill is compute-bound; decode is bandwidth-bound” as a hypothesis that depends on workload and hardware.

**Evidence:** two small curves/tables and a paragraph explaining exceptions. **Stop:** you can predict the next workload point qualitatively.

### L2-B — E03: find a useful capacity envelope

**Consequence:** distinguishes high throughput from an acceptable user experience.

- [ ] **L2-B01** Define a provisional workload-specific SLO before the sweep: acceptable TTFT, time-per-output-token or completion latency, and error rate. Label it a lab target, not a universal service standard.
- [ ] **L2-B02** Sweep concurrency 1, 2, 4, 8 and, only if useful, 16 at a fixed shape. Stop after a clear saturation/latency boundary or memory pressure.
- [ ] **L2-B03** Record aggregate output tokens/sec, request rate, per-request latency, running/waiting requests and cache pressure. More occupied memory or higher GPU utilization alone does not prove efficiency.
- [ ] **L2-B04** At the chosen operating point, run a modest offered-arrival-rate test as well as closed-loop concurrency. Verify the client can generate the requested load and count offered, admitted, completed and failed requests separately.
- [ ] **L2-B05** Name the best operating point for your target and explain what saturates first. Show the point where adding load mainly increases queueing.

**Evidence:** capacity table with SLO pass/fail and a load-generation description. **Stop:** no exhaustive search over every request rate.

### L2-C — E04: test one scheduler or memory decision

**Consequence:** makes tuning deliberate rather than a collection of copied flags.

- [ ] **L2-C01** Read the selected engine's definitions of maximum context, sequence concurrency, token budget and GPU-memory budget. Identify the knob matching your measured bottleneck.
- [ ] **L2-C02** Choose one knob, record a baseline and two alternatives, and predict the latency/throughput/memory tradeoff. For example, change the batched-token budget under a mixed short/long-prompt workload.
- [ ] **L2-C03** Run the controlled comparison without changing precision, GPU, reasoning or workload distribution at the same time. Record effective settings from logs.
- [ ] **L2-C04** Inspect long-prefill interference with ongoing decode, or cache/preemption behaviour if that was the target. Use server metrics where exposed.
- [ ] **L2-C05** Adopt or reject the change. Recheck representative output correctness. If no gain appears, explain why and retain the baseline; investigate a second knob only if a concrete bottleneck remains.

**Evidence:** a before/after decision with the cost of the tradeoff. Reference: [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).

### L2-D — E05: connect reasoning policy to serving economics

**Consequence:** avoids optimizing raw token speed while making task completion worse.

- [ ] **L2-D01** Build a reusable 20-prompt sanity set: ten simple tasks and ten harder tasks, each with an answer check or a written scoring rubric. Include a few relevant infrastructure questions with independently checked answers.
- [ ] **L2-D02** Compare thinking disabled with one explicit supported reasoning effort. Keep the token budget adequate for the task and log truncation; save full generation settings.
- [ ] **L2-D03** Record first-output time, first-answer time where observable, completion time, generated tokens and success by prompt ID.
- [ ] **L2-D04** Investigate failures before calling the faster policy superior. Separate wrong answers, malformed output, exhausted budgets and server errors. Repeat ambiguous sampled cases only where the decision depends on them.
- [ ] **L2-D05** Recommend a policy for each workload class. State that a 20-case set is a regression check and learning tool, not a general model-quality ranking.

**Evidence:** paired task results and a policy choice. If model-card sampling recommendations differ between thinking modes, either test those complete policies and label the confound, or hold sampling fixed for an isolated experiment.

### L2-E — Diagnose with telemetry and a short trace

**Consequence:** builds the performance-analysis skill repeatedly relevant to NVIDIA work.

- [ ] **L2-E01** Capture timestamps alongside GPU memory/utilization, clocks/power where available, CPU usage, host RAM and engine running/waiting counts during one experiment.
- [ ] **L2-E02** Correlate a slow interval with telemetry. Propose two possible explanations and identify the observation that would distinguish them.
- [ ] **L2-E03** Capture a short steady-state Nsight Systems trace, or an engine/PyTorch trace if profiling access is unavailable. Focus on a handful of requests rather than the full model download/startup.
- [ ] **L2-E04** Identify GPU execution, CPU launch work, idle gaps and memory copies where visible. A trace can suggest bandwidth limits; proving them may need counters or a quantitative model.
- [ ] **L2-E05** Write one evidence-backed diagnosis and one limitation. If traces are blocked by the host, retain the error and use telemetry; mark the Nsight skill as pending rather than blocking the whole model study.

**Evidence:** annotated trace or telemetry interval and diagnosis. References: [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics/).

### L2-F — E06: distinguish network and serving delays

**Consequence:** turns your networking experience into inference-specific evidence.

- [ ] **L2-F01** Run the same light workload from a client near the server and from your normal remote client, using the same client version and request policy. Record route/proxy differences and client resources.
- [ ] **L2-F02** Measure connection setup separately from warm keep-alive requests where possible. Compare TTFT, total latency and streaming chunk arrival patterns.
- [ ] **L2-F03** Check whether proxy buffering, connection reuse, timeouts or client parsing could explain the difference. Use server-side timing to bound inference time when exposed.
- [ ] **L2-F04** Estimate request/response byte volume for the text workload. Explain why WAN request traffic is a different problem from GPU collectives and state transfer between serving workers.
- [ ] **L2-F05** Write a deployment implication: for example, use a regional client, preserve streaming, fix a timeout or move the load generator. Do not attribute the entire location difference to propagation delay.

**Evidence:** local/remote comparison and one justified network decision. Streaming does not require a fresh full RTT for every generated token. A TCP throughput test alone does not measure LLM response latency.

### L2-G — E07: practice one service failure and recovery

**Consequence:** tests whether the deployment is operable beyond the happy path.

- [ ] **L2-G01** Define readiness, a per-request timeout and bounded admission/concurrency. Record how a client sees model loading versus a ready service.
- [ ] **L2-G02** Send an over-limit request in the lab. Check rejection or truncation behaviour and verify that a normal request still works afterwards.
- [ ] **L2-G03** Cancel or time out one long request. Observe whether active-request/cache pressure recovers; do not assume a disconnected client automatically stops GPU work.
- [ ] **L2-G04** Restart the service with model files cached. Measure time to readiness and first useful response; separate this from the uncached startup.
- [ ] **L2-G05** Write a small runbook: symptom, first three checks, recovery action and validation request. Choose a real failure or a clearly labelled injected one.

**Evidence:** a recovery timeline and a usable runbook. **Stop:** one bounded lab drill is sufficient; no need to build HA infrastructure yet.

### L2-H — E08: convert the measurements into a hosting choice

**Consequence:** connects engineering work to consultancy and solution architecture.

- [ ] **L2-H01** Record actual GPU rental, storage and other material costs for the experiment window. Distinguish observed charges from quoted rates.
- [ ] **L2-H02** Calculate cost per million output tokens for the fixed benchmark, and cost per successful task for the sanity set. Include failures/retries in cost; define both denominators.
- [ ] **L2-H03** Compare the baseline and chosen operating configuration at the same quality/latency requirement. Label any different hardware, precision or input distribution explicitly.
- [ ] **L2-H04** Estimate what changes at low utilization, longer prompts and a burst of demand. State which estimates need measurement before a customer commitment.
- [ ] **L2-H05** Write a one-page recommendation: workload, deployment, measured capacity, latency, costs, failure behaviour, limits and next experiment.

**Evidence:** an engineering recommendation someone else can challenge and reproduce. A higher peak tokens/sec result is not automatically the cheapest useful service.

### L2-I — Make the result transferable

**Consequence:** prevents starting from zero on model 02.

- [ ] **L2-I01** Preserve the workload IDs, benchmark runner, metric dictionary, manifest format and decision template as reusable assets.
- [ ] **L2-I02** Update the shared concepts register in section 11. Mark each concept understood, measured or transferred; do not mark transfer before testing another model.
- [ ] **L2-I03** Record three predictions for model 02: one about memory, one about performance and one about a distinctive mechanism. Verify that model's actual configuration before relying on family names.
- [ ] **L2-I04** Separate observations specific to Qwen/GPU/runtime from conclusions likely to generalize. Give each generalization a condition or counterexample.
- [ ] **L2-I05** Pass G2 and stop expanding this model's core scope. Select the next model while retaining the same harness where valid.

**Evidence:** a portable experiment pack and a next-model handoff.

### Gate G2 — Enough to move to the next model

- [ ] **G2-01** G1 still passes with the final pinned configuration.
- [ ] **G2-02** Input, output and load sweeps explain the useful operating envelope.
- [ ] **G2-03** One controlled optimization attempt has a defensible adopt/reject decision and a correctness check. A positive speedup is not required.
- [ ] **G2-04** Reasoning/output length is included in the task-cost interpretation.
- [ ] **G2-05** One bottleneck diagnosis uses telemetry or traces, with uncertainty stated.
- [ ] **G2-06** I have compared the client/network path and demonstrated a bounded recovery procedure.
- [ ] **G2-07** A short hosting recommendation and reproducibility pack exist.
- [ ] **G2-08** I can name what the next model will teach or test.

**When these pass, move on.** An unavailable optional profiler may be documented without blocking G2 if a meaningful telemetry-based diagnosis exists. An untested core experiment remains incomplete; a hardware blocker is a reason for a deliberate detour, not a completion claim.

## 6. Level 02 extensions — choose only with a reason

These are valuable, especially for NVIDIA-oriented work. Complete at most one before changing models unless it solves an immediate problem. An extension can replace the optimization experiment in L2-C if it tests the same kind of controlled deployment decision; retain L2-C's evidence and correctness requirements.

### L2-XQ — Quantization with a quality and hardware check

**Choose when:** memory is limiting, or a cheaper deployment is plausible.

- [ ] **L2-XQ01** Select one supported quantized derivative and inspect its base revision, format, scale scheme and retained higher-precision tensors. Record whether weights, activations or KV are quantized.
- [ ] **L2-XQ02** Verify that the chosen GPU/runtime has a supported execution path. Loading a format does not prove accelerated kernels are used.
- [ ] **L2-XQ03** Compare memory, startup, latency and throughput at the same workload and concurrency. Keep KV precision fixed for a weight-only comparison; flag any unavoidable change in GPU or base revision.
- [ ] **L2-XQ04** Run the paired sanity set and retain changed answers. Decide whether capacity/cost improves enough to justify the observed quality risk.

**Enough:** one credible precision tradeoff, not a survey of all quantizers. If both variants cannot fit on the same hardware, call it a deployment comparison, not an isolated precision result.

### L2-XC — Prefix reuse in a hybrid model

**Choose when:** repeated system prompts, documents or conversation prefixes are material.

- [ ] **L2-XC01** Check prefix-caching support for this exact hybrid model and engine revision, including recurrent-state behaviour. Save the relevant support note or observed limitation.
- [ ] **L2-XC02** Create unique-prefix, cold shared-prefix and warm shared-prefix cases with identical output policies. Reset or intentionally prime caches between cases.
- [ ] **L2-XC03** Compare TTFT, server cache-hit evidence and outputs. Check reuse at a different prefix boundary if the engine has block-alignment constraints.
- [ ] **L2-XC04** State what was reused and what was recomputed. If unsupported, document the gap and defer implementation work to L3-H.

**Enough:** measured reuse, or a verified support limitation. The [hybrid KV cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) shows why generic full-attention caching assumptions need care; the checked page still describes some Mamba-related support as work in progress. Model-specific runtime evidence takes priority over assuming blanket support or blanket absence.

### L2-XT — Two GPUs: a topology and communication bridge

**Choose when:** you have an affordable two-GPU window and want direct evidence for infrastructure roles. High priority for a later return; not required now.

- [ ] **L2-XT01** Record the physical/virtual topology, GPU form factor, PCIe/NVLink links, CPU affinity and peer-access availability. Two GPUs in one VM do not guarantee a fast peer path.
- [ ] **L2-XT02** Run an appropriate peer-bandwidth/latency check and a small `nccl-tests` sweep; verify correctness. Save NCCL version, topology and logs showing the transport actually selected.
- [ ] **L2-XT03** Compare TP=1 with TP=2 only if the same model/precision fits the one-GPU case. Record per-GPU memory, latency, throughput and total cost. Otherwise label it a capacity-enabling deployment comparison.
- [ ] **L2-XT04** If each GPU can independently fit the model, compare TP=2 against two TP=1 replicas at the same total GPU count and offered workload. Explain the tradeoff between splitting computation and serving independent requests.

**Enough:** explain one measured scaling result. Do not claim NCCL performance if the engine uses a different custom collective path. References: [NCCL tests](https://github.com/NVIDIA/nccl-tests), [NCCL diagnostics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).

### L2-XK — GPU Kubernetes deployment

**Choose when:** an existing GPU cluster is available or a target role specifically requires it.

- [ ] **L2-XK01** Explain the device plugin, GPU Operator, driver/container runtime and scheduler responsibilities. Map a requested GPU resource to a visible device inside the pod.
- [ ] **L2-XK02** Deploy the known working configuration with an explicit GPU request, persistent model cache, startup/readiness probes and request limits. Confirm the image and model revision match the baseline.
- [ ] **L2-XK03** Run a smoke load and restart the pod. Observe scheduling, image pull, model loading and readiness separately.
- [ ] **L2-XK04** Correlate pod/request behaviour with GPU/engine telemetry. Write one failure diagnosis such as an unavailable GPU, image mismatch or inappropriate startup probe.

**Enough:** one reproducible GPU workload and one recovery explanation. Use an actual GPU worker; a CPU-only local Kubernetes cluster does not validate GPU serving. Reference: [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html).

## 7. level03-depth — return when a concrete question demands it

Do not work through this section sequentially. Pick one module, write the question and the expected artifact, and stop when its exit criterion is satisfied. A single model is not expected to support every feature below.

### L3-H — Hybrid attention and state management

**Trigger:** memory scaling, prefix reuse or a state-management bug is unexplained.

- [ ] **L3-H01** Trace the runtime's DeltaNet and attention state allocation from configuration to tensor shapes. Separate conceptual state from reserved engine pools.
- [ ] **L3-H02** Estimate recurrent-state bytes from the actual layer/head dimensions and storage dtype, including convolution state. Compare with runtime tensor inspection where accessible.
- [ ] **L3-H03** Locate prefill and decode paths, identifying how recurrent state is produced, updated and retained. Explain why parallel prefill need not execute like sequential decode.
- [ ] **L3-H04** Test one state boundary: prefix reuse, cancellation, preemption or checkpointing. Compare output/correctness against a clean run.
- [ ] **L3-H05** Produce an annotated explanation or minimal bug reproducer with exact versions and conditions.

**Enough:** one state-management question resolved or isolated to a specific implementation limit. No need to recreate the model from scratch.

### L3-P — GPU profiling, arithmetic intensity and kernels

**Trigger:** a timeline identifies an expensive operation or unexplained GPU idle period.

- [ ] **L3-P01** Form a quantitative bottleneck hypothesis: approximate operations and bytes moved for the selected shape. Include assumptions about weight reuse and cache traffic.
- [ ] **L3-P02** Use Nsight Systems to isolate the operation and its surrounding CPU/GPU work; distinguish startup/compilation from steady state.
- [ ] **L3-P03** If appropriate and permitted, use Nsight Compute on the selected kernel to inspect memory traffic, occupancy and execution efficiency. Record profiler overhead and sampling limits.
- [ ] **L3-P04** Evaluate one supported change such as CUDA graph mode, batch shape or backend selection. Confirm what kernel/backend ran instead of inferring it from a flag.
- [ ] **L3-P05** Only if necessary, construct a small PyTorch/Triton/CUDA reproducer. Check numerical agreement, benchmark properly and explain the end-to-end impact ceiling.

**Enough:** measured evidence for one bottleneck and a bounded improvement or negative result. Kernel coding is an advanced route, not a prerequisite for an inference solutions-architect portfolio. Start with the [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

### L3-S — Speculative decoding and MTP

**Trigger:** low-concurrency decode latency dominates and a compatible drafting path exists.

- [ ] **L3-S01** Verify draft weights, runtime support, precision compatibility and correctness guarantees for the selected method. Distinguish a trained MTP head from enabled runtime speculation.
- [ ] **L3-S02** Compare drafting off with one small speculative-token setting under a fixed workload and reasoning policy.
- [ ] **L3-S03** Record accepted/drafted tokens, extra memory, total latency and throughput at concurrency 1 and at a loaded operating point.
- [ ] **L3-S04** Explain whether draft generation, verification or rejected tokens dominate. Higher acceptance alone does not prove a speedup.
- [ ] **L3-S05** Keep or disable speculation using measured benefit and output checks. Preserve a plain decoding fallback.

**Enough:** establish the workload conditions where speculation helps or hurts. Use the hardware-specific [Qwen vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) as a starting point; do not transplant its reported acceptance or throughput.

### L3-N — Multi-node GPU networking and collectives

**Trigger:** a distributed model deployment, NVIDIA networking opportunity or a real communication bottleneck requires it.

- [ ] **L3-N01** Confirm actual RDMA-capable NIC access, fabric connectivity, driver support and provider permissions before renting multiple nodes. If unavailable, design the lab and mark execution pending.
- [ ] **L3-N02** Inventory GPU–NIC/CPU topology, NIC link rate, NUMA placement, routing and interface names. Distinguish NVLink inside a system from InfiniBand/RoCE between systems.
- [ ] **L3-N03** Run fabric-appropriate latency/bandwidth checks and an NCCL message-size sweep with correctness validation. Do not equate TCP `iperf` bandwidth with RDMA or collective performance.
- [ ] **L3-N04** Confirm selected transport from logs and counters. Check for unexpected socket fallback, wrong interfaces, poor affinity or disabled GPU peer access before changing tuning variables.
- [ ] **L3-N05** Relate collective performance to inference step time using a distributed workload or trace. Treat all-reduce, all-gather and all-to-all as different traffic patterns.
- [ ] **L3-N06** On an isolated fabric you control, investigate a single congestion/placement issue; for RoCE, consider the actual ECN/PFC policy and counters. Do not modify shared network policy to complete a study task.

**Enough:** explain one end-to-end distributed inference result with topology, transport and communication evidence. Qwen's dense backbone can teach tensor-parallel collectives; expert routing/all-to-all belongs with an appropriate MoE model. References: [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html), [NCCL tests](https://github.com/NVIDIA/nccl-tests).

### L3-D — Disaggregated prefill/decode and state movement

**Trigger:** long prefills disrupt decode, or a multi-worker architecture decision needs evidence.

- [ ] **L3-D01** Verify the serving framework/connector supports the exact hybrid model and transfers all required state. Full-attention-only examples do not prove DeltaNet support.
- [ ] **L3-D02** Estimate attention-cache and recurrent-state transfer bytes per handoff. Divide by achievable, measured link bandwidth for an ideal transfer lower bound, then account for protocol and scheduling costs.
- [ ] **L3-D03** Compare colocated and disaggregated serving at equal total hardware and a fixed workload. Include routing, serialization, transfer and queue time.
- [ ] **L3-D04** Test the relevant failure/retry path and observe whether a failed transfer causes recomputation or request failure.
- [ ] **L3-D05** Recommend when disaggregation earns back its communication and operational cost. If support is absent, select another model for the experiment and preserve the architectural question.

**Enough:** a bounded break-even analysis with measured inputs, or a verified compatibility limitation. This can become an NVIDIA Dynamo/NIXL study later; product support must be checked when selected.

### L3-O — Production orchestration and reliability

**Trigger:** a customer, role exercise or sustained service requires Kubernetes and multiple workers.

- [ ] **L3-O01** Turn the known workload into a deployment spec covering GPU placement, persistent storage, secrets, startup/readiness and graceful shutdown.
- [ ] **L3-O02** Establish correlated engine, GPU and node metrics; choose alerts tied to user impact rather than GPU utilization alone.
- [ ] **L3-O03** Test overload and bounded admission with realistic arrival patterns. Measure useful completions, latency and rejected demand.
- [ ] **L3-O04** Test one rollout, worker failure or scale event. Account for cold starts, loading time and request draining.
- [ ] **L3-O05** Evaluate one orchestration choice, such as replica placement or an autoscaling signal, with measured data.

**Enough:** one operational design and failure exercise a reviewer can inspect. Do not claim production reliability from a short lab. Reference: [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html).

### L3-V — Vision and long-context serving

**Trigger:** a real workload needs images or contexts beyond your baseline.

- [ ] **L3-V01** Add one controlled image workload; record source resolution, preprocessing, visual-token accounting and encoder execution.
- [ ] **L3-V02** Measure upload/preprocessing/encoder/prefill contributions where observable. Record processor settings so the same image gives a comparable workload.
- [ ] **L3-V03** Extend text context in safe steps with a bounded output budget. Measure cache/state pressure, latency and task quality at each step.
- [ ] **L3-V04** Compare meaningful retrieval/reasoning tasks at those lengths. A successful allocation or advertised context length does not demonstrate effective long-context quality.

**Enough:** a supported context/modality envelope for the chosen application. Context extension beyond the native limit is a separate compatibility and quality experiment.

### L3-T — Post-training through its inference consequences

**Trigger:** the model misses a specific task requirement, or behaviour creates avoidable serving cost.

- [ ] **L3-T01** Define the target behaviour, held-out evaluation set and acceptable serving budget before considering SFT, preference tuning or reinforcement learning.
- [ ] **L3-T02** Establish whether prompting, constrained output or reasoning policy already solves the problem. Record the residual failure that justifies training.
- [ ] **L3-T03** If an adapter experiment is justified, compare base, adapter-served and merged deployment where supported. Measure memory, throughput, correctness and output-length changes.
- [ ] **L3-T04** For RL-style work, estimate rollout-generation cost and model-update/serving compatibility. Treat the training pipeline as its own project with a separate budget.
- [ ] **L3-T05** Accept customization only if held-out task gains justify training and serving costs. Track task success and completion time together.

**Enough:** one validated behaviour change and its serving impact. Full RLHF/RL training is not a gate for moving beyond this first inference model.

### Gate G3 — Enough depth for the selected module

- [ ] **G3-01** The original question and its deployment consequence are written down.
- [ ] **G3-02** The explanation is supported by a trace, code inspection, controlled measurement or a verified support limitation.
- [ ] **G3-03** Another engineer can reproduce the key observation from the saved setup.
- [ ] **G3-04** The conclusion states scope, uncertainty and what would trigger a revisit.

Close the module and move on. Completing every Level 03 module is not a meaningful gate.

## 8. Experiment catalog and measurement rules

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

## 9. NVIDIA role alignment — where to double down

The priority is to demonstrate infrastructure and performance judgement with hands-on evidence. Your existing networking experience becomes more distinctive when connected to GPU placement, communication, serving latency and cost.

The following is a skill-signal sample from NVIDIA's official career listings checked on the source date. Several job pages render dynamically; some requirements were available through indexed official excerpts rather than the full description. This is not a complete hiring-criteria audit or a claim that these openings are London-based or will remain open.

| Official role signal | Observed emphasis | Checklist priority and proof |
|---|---|---|
| [Solutions Architect, Inference Deployments](https://jobs.nvidia.com/careers/job/893393730455) | Deploying and improving inference at scale; GPU technology and Kubernetes | **Core:** reproducible server, E03/E04 capacity/tuning, E07 recovery, E08 recommendation. **Return:** L2-XK. |
| [Senior Software Engineer, AI Inference Systems — JR2008495](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Senior-Software-Engineer--AI-Inference-Systems_JR2008495) | CUDA/memory hierarchy/streams/NCCL; Nsight; Python and C/C++ | **Core:** L2-E profiling and benchmark automation. **Return:** L3-P, L2-XT and L3-N; deeper programming for the software-engineering track. |
| [Solutions Architect, Networking — JR2017665](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Solutions-Architect--Networking_JR2017665) | Designing and deploying large AI-factory networks | **Core:** L1-G topology and L2-F network attribution. **Return:** topology/collectives evidence in L2-XT and L3-N. |
| [Accelerated Kubernetes Performance and Scale — JR2020019](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Senior-Systems-Software-Engineer--Accelerated-Kubernetes-Performance-and-Scale---DGX-Cloud_JR2020019) | GPU operators, device plugins and distributed inference serving | **Return:** L2-XK followed by a bounded L3-O operational experiment. |

### Recommended emphasis for your background

1. **Measurement and diagnosis:** per-request data, queueing, traces, GPU memory and bottleneck attribution.
2. **GPU systems and topology:** memory hierarchy, CPU/NUMA placement, PCIe/NVLink and the path to the NIC.
3. **Distributed communication:** collective correctness, message-size sensitivity, transport verification and cost of scaling.
4. **Operating deployments:** Linux, containers, restart/readiness, admission, observability and then GPU Kubernetes.
5. **Customer-facing engineering:** state the workload, constraints, alternatives, evidence and recommendation clearly.
6. **Code fluency:** maintain a readable Python harness now; deepen C++/CUDA selectively for software/performance-engineer targets.

For your broader role preparation, the advanced infrastructure items are deferred, not disposable. After establishing the first two model baselines, prioritize a real two-GPU comparison (L2-XT), a focused GPU trace (L2-E/L3-P), and one GPU Kubernetes deployment (L2-XK). Add multi-node RDMA work when suitable fabric access exists. These are evidence targets for the wider program, not extra conditions on G2.

These are my curriculum priorities inferred from the role signals and your background, not a claim that every NVIDIA role requires the same stack. SA/infrastructure roles and CUDA software-engineer roles have different depth requirements. Do not make kernel development a prerequisite for pursuing the former.

### Keep the NVIDIA stack in view without turning it into a product checklist

| Layer | Tools/products to recognize | Evidence to seek |
|---|---|---|
| Execution | CUDA and TensorRT-LLM; vLLM as the first engine | Which runtime/backend executes the selected model and why |
| Packaging/deployment | Containers and NVIDIA NIM where applicable | Supported model/hardware profile, configuration and operational behaviour |
| Distributed serving | NVIDIA Dynamo and state-transfer components such as NIXL | Routing/state movement costs and exact compatibility |
| GPU orchestration | GPU Operator and device plugins | GPU scheduling, visibility, readiness and recovery |
| Observability | Nsight Systems/Compute; DCGM where available | Correlated measurements that explain user-visible latency or failures |
| Communication | NCCL, NVLink, InfiniBand and RoCE | Actual transport/topology and measured communication behaviour |

This is an orientation map, not a claim that Qwen3.8-27B is supported in every listed product. Check the exact support matrix before selecting a lab. For engine orientation, see [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html).

## 10. Evidence pack — small enough to maintain

Suggested future repository locations; these are organizational recommendations, not files already created by this checklist:

| Location | Contents |
|---|---|
| `models/qwen3.8-27b/README.md` | Current deployment, findings, gates and links |
| `models/qwen3.8-27b/configs/` | Pinned manifests and launch configurations |
| `models/qwen3.8-27b/results/` | Raw results and concise comparisons |
| `models/qwen3.8-27b/profiles/` | Selected traces or links to large captures |
| `concepts/` | Shared explanations with links to supporting experiments |
| `benchmarks/` | Reusable workloads and benchmark runner |
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
max_model_len: 16384
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

## 11. Shared concepts register

Use **U = understood**, **M = measured**, **T = transferred to another model**. Leave a state blank until demonstrated. “Transferred” means using the concept to predict or diagnose the new model, including discovering that an old assumption fails.

| Concept | U | M | T | First evidence / later question |
|---|---|---|---|---|
| Weights versus runtime memory | — | — | — | L1-B; do parameter labels predict loaded bytes? |
| KV cache, GQA and recurrent state | — | — | — | L1-B/G; L3-H if needed |
| Prefill versus decode | — | — | — | E01/E02 |
| HBM bandwidth versus compute | — | — | — | L2-E; L3-P for proof |
| Continuous batching and queueing | — | — | — | E03 |
| Chunked prefill and scheduling budgets | — | — | — | E04 if selected |
| Reasoning length and useful task cost | — | — | — | E05/E08 |
| Quantization and execution kernels | — | — | — | L1-B; L2-XQ when selected |
| Prefix reuse and cache correctness | — | — | — | L2-XC when supported |
| CUDA graphs and host launch overhead | — | — | — | L2-E; L3-P |
| Network path versus GPU service time | — | — | — | E06 |
| GPU/CPU/NIC topology and NUMA | — | — | — | L1-G; L2-XT |
| Tensor parallelism versus replicas | — | — | — | L2-XT |
| NCCL/collectives and RDMA transport | — | — | — | L2-XT; L3-N |
| Speculative decoding acceptance and cost | — | — | — | L3-S |
| Prefill/decode state handoff | — | — | — | L3-D |
| Readiness, overload and recovery | — | — | — | E07; L3-O |
| Capacity planning and goodput | — | — | — | E03/E08 |

**Reuse rule:** on each later model, rerun a compact baseline and one shared diagnostic experiment. Spend most new learning time on that model's distinctive mechanism. Do not replay every completed Qwen task unless assumptions or behaviour changed.

## 12. Parking lot and return rules

| Question | Why it matters | Evidence needed | Return trigger | Status |
|---|---|---|---|---|
| Example: why does long-context latency jump? | May change the supported workload | Trace plus cache/preemption metrics | A repeated discontinuity in E01 | Unstarted |
| Example: would TP=2 beat two replicas? | GPU/network architecture choice | Same-hardware comparison | Two suitable GPUs available | Unstarted |
| Example: will a smaller dtype help? | Capacity and rental cost | Paired runtime/quality comparison | Weight or cache memory limits E03 | Unstarted |

Return when the answer could change a deployment decision, explain a repeated anomaly, meet a concrete role requirement or support a useful contribution. Curiosity is welcome; record it without making it a hidden prerequisite.

**Final stopping rule:** after G2, you should be able to say: “For this workload, on this hardware and software revision, I can reproduce the service, explain the main bottleneck, defend the operating configuration and state its limits.” That is enough for model 01.

## 13. Primary references

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

The NVIDIA career references and their evidence limitations are in section 9. Numerical memory examples are calculations from published dimensions; workload sizes, timeboxes, tasks and progression criteria are proposed learning-design choices, not benchmark claims from those sources.
