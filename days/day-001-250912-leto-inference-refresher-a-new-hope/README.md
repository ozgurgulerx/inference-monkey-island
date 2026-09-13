---
id: day-001
title: "Leto — Inference refresher: A New Hope"
day: 1
date: "2025-09-12"
items: []
---

I am back to focusing on inference subjects...
With my departure from MS, feeling pumped up as the formal detachment from the "copilot" will give me the much sought after focus on deeply tech subjects on LLM's. 

I previously covered inference basics with the books like the one from baseten and with my own study run  previously documented here [inference journal](https://github.com/ozgurgulerx/inference-journal)

This time I will "work backwards" from the edge of the field to stay relevant and upto date not wasting time on the basics while building more depth into the core inference subjects. 


The model of the day is DeepSeek V4.1 Flash, GLM-5.3, Kimi K3, Nemotron 3.5 Lightning.
Inkling from Thinking Machines... 
All major neoclouds are announcing they are starting serving the model. 


## Class 1: Core model families

| Model / family | Architecture and differentiation | Serving and performance focus | Expected practical depth |
| --- | --- | --- | --- |
| **Qwen3.8-27B** | Dense feed-forward computation with Gated DeltaNet and attention layers, plus native vision. Useful for understanding contemporary hybrid models. | Compare recurrent-state and attention-cache costs; tune batching, prefill scheduling and quantization. Measure image preprocessing separately. | **Main current-model lab.** Start with text, then introduce vision. |
| **GLM-4.7-Flash** | 30B total / ~3B active MoE. Small enough to make sparse-model experiments accessible. | Expert routing, grouped expert computation, weight bandwidth, tensor parallelism and expert parallelism. | **Main small-MoE lab.** Single GPU first, then controlled multi-GPU experiments. |
| **GLM-5.3-Flash** | 320B total / 18B active, hybrid sparse/linear attention, native multimodality and mHC residual connections. | Large-model memory accounting, hybrid-state handling, precision choices and architecture-specific runtime support. | **Master the architecture.** Full deployment comes later; GLM-4.7-Flash teaches some MoE fundamentals but is not an architectural substitute. |
| **DeepSeek lineage → V4.1 Flash** | Latest release combines causal encoder-decoder organization, compressed sparse attention, conditional memory and speculative decoding. | Cache compression, memory placement, different prefill/decode requirements and communication costs. | **Core distributed-systems case study.** Reproduce components and smaller experiments before full deployment. |
| **Nemotron 3.5 Lightning** | Mamba/Transformer hybrid MoE, multi-token prediction and an NVFP4-oriented deployment path. | Recurrent-state management, hybrid batching, low-precision execution and speculation. | **Second architecture lab** after dense and small-MoE serving are familiar. |

**Architecture and deployment references:** Qwen, GLM-4.7-Flash, GLM-5.3-Flash, DeepSeek, Nemotron.

**Baseline:** Add one conventional small dense transformer as a baseline. Learn ordinary attention and KV-cache growth before tackling hybrid designs. That makes architectural improvements measurable and understandable.

## Class 2: Important production designs and comparisons

| Model | Why study it | What deserves attention |
| --- | --- | --- |
| **Kimi K3** | Giant sparse-model serving, with Kimi Delta Attention and Attention Residuals. | Expert placement, topology, communication overlap, long-context state and multi-node reliability. High conceptual relevance to your networking background; lower immediate hands-on priority. |
| **GLM-5.3 / GLM-5.2** | Useful production examples with published provider optimization work. | Study Baseten’s combination of quantization, KV-aware routing, disaggregation and multi-token prediction. Distinguish improvements to the model from improvements to its serving system. |
| **Qwen3.8-Max / 2.4T-A95B** | Another giant MoE reference. | Compare its resource requirements with Kimi. One detailed comparative exercise is sufficient initially. |
| **Qwen3.8-Flash-Next** | Experimental architectural direction. | Identify changes that require new kernels, memory layouts or scheduler behaviour. |
| **Inkling** | Native multimodal serving reference. | Modality-specific processing, request variability and scheduling—particularly if multimodal inference becomes a commercial focus. |

**Sources:** Kimi, Baseten’s GLM serving design, Qwen Max listing, Flash-Next, Inkling.


## My approach

I want to work backwards from what is happening in inference now, then go deeper into the systems questions behind it. The models above give me concrete cases to work through. I want to understand what changes in serving when the architecture changes, and test that understanding in labs.

I will start with a few architectures and build depth across them. For each experiment, I want to predict the bottleneck before running it, measure what actually happens, then explain the result in my own words before asking AI for hints or review.

### Performance techniques I want to master

Priority 1 is the core work I want to repeat across the models. Priority 2 covers techniques I will investigate when the workload gives me a reason to use them.

| Priority | Area | What I will experiment with | What I want to be able to explain |
| --- | --- | --- | --- |
| 1 | Profiling and bottleneck analysis | Separate prefill, decode, CPU work, memory traffic and communication. | Why a configuration is slow, before I start changing it. |
| 1 | Batching and scheduling | Arrival rate, concurrency, sequence lengths, chunked prefill and admission control. | How far I can raise throughput while keeping request latency within the target. |
| 1 | Memory accounting | Weights, KV cache, recurrent state, activations, communication buffers and runtime overhead. | How much useful serving capacity I have once everything is accounted for. |
| 1 | Quantization | Weight, activation and cache precision separately; supported kernels; quality regression. | Whether the gain comes from arithmetic, bandwidth or room for larger batches, and what happens to quality. |
| 1 | Parallelism | Replicas versus tensor parallelism; expert parallelism; pipeline parallelism when justified. | How computation and communication map onto the actual topology. |
| 1 | Reliability | Bursts, cancellation, OOM, worker failure, cold starts and overload. | How the service behaves under stress, where it fails and how it recovers. |
| 2 | Prefix caching and routing | Cache hits, eviction, replica locality and load imbalance. | Whether saved prefill work outweighs the extra queueing delay. |
| 2 | Speculative decoding | Draft cost, acceptance, verification cost and concurrency. | When speculation reduces latency and when it wastes capacity. |
| 2 | Prefill/decode disaggregation | Phase interference, independent scaling and state-transfer overhead. | When separating the phases pays for itself. |

These optimizations depend on the workload. I want to keep that in view when reading benchmark claims. The vLLM guidance describes speculation as useful for memory-bound workloads at medium-to-low QPS; Dynamo distinguishes disaggregation from cache-aware routing. I will treat those as starting points for experiments and check the conditions against the runtime version I use.

References to follow: vLLM tuning, expert parallelism and speculation; NVIDIA Dynamo.

### Technical niches I want to investigate

#### 1. Communication-aware inference

I want to connect collective communication, expert dispatch and cache transfer to request latency and serving capacity. This is where I want to bring my networking background into the inference work. I will look at topology, message sizes, synchronization, congestion and communication/computation overlap. Each of those traffic patterns needs its own analysis.

References to follow: parallelism and KV transfer.

#### 2. Hybrid-state management

Recurrent state and attention KV have different storage and reuse properties. I want to investigate what that means for prefix reuse, branching, checkpointing, eviction and speculative rollback. I will check what the runtime actually supports for each model implementation.

References to follow: Qwen and Nemotron architecture documentation.

#### 3. Architecture-dependent infrastructure economics

The DeepSeek V4.1 model card reports different activated computation during prefill and decode, along with compressed cache representations. My working hypothesis is that these changes can alter the useful prefill/decode worker ratio and the network cost of disaggregation. I want to test that. An architecture update may mean I have to rebuild the capacity model rather than reuse the old assumptions.

Reference to follow: DeepSeek model card.

#### 4. Reasoning budgets and output length

The usage notes for GLM-5.3-Flash and Kimi K3 describe maximum reasoning effort as the default. I want to benchmark time to a useful answer, task success and total resource consumption at different reasoning budgets. Decode tokens/sec alone can hide a much slower completed task.

References to follow: GLM and Kimi usage notes.

#### 5. Faithful model onboarding

I want correctness testing to include chat encoding, tool-call parsing, reasoning history and image processing. DeepSeek V4.1 supplies custom prompt encoding rather than a Jinja template, so I need to understand how the integration handles it. Fluent output is only one part of checking that the model has been onboarded correctly.

Reference to follow: encoding documentation.

### Providers and projects I will use as study sources

| Class | Providers / projects | What I will study through them |
| --- | --- | --- |
| 1 | vLLM, SGLang, NVIDIA Dynamo; Baseten, Together, Fireworks | Runtime internals and concrete serving optimizations. |
| 2 | Runpod, Modal, Nebius, CoreWeave | Deployment, autoscaling, orchestration, topology and capacity operations. |
| 2 | Cerebras, Groq, SambaNova | Alternative hardware designs and their trade-offs. |
| 3 | DeepInfra, Novita, Parasail, FriendliAI, SiliconFlow, GMI, Chutes | Model adoption and commercial demand. I will pick out technically substantive articles individually. |

For the labs, I plan to start with vLLM, reproduce selected results in SGLang, then bring in Dynamo when I study routing and disaggregation. I will choose the subject as I go; this gives me a starting point for the tooling.

My test of progress will be concrete: predict the bottleneck, measure it, improve it, and explain where the improvement stops working. I want to build that evidence across a few architectures. At the end of each study day, I will take the interview perspective: explain the concept, apply it to a serving scenario, and defend what my measurements support.

