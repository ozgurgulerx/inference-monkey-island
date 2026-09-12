---
id: day-001
title: Leto — Inference refresher: A New Hope
day: 1
date: "2026-09-12"
items: []
---

# Day 01 — Leto: Inference refresher — A New Hope

## What I am trying to achieve today

Get back up to speed on the basics and most important subjects in inference engineering. Rebuild a working picture of how an LLM request is served, identify gaps in my understanding, and choose the next subject to explore.

### Suggested refresher scope

- **Request lifecycle and model basics:** tokenization, transformer forward pass, attention, autoregressive generation, prefill and decode.
- **Memory and hardware constraints:** model weights, precision, KV cache growth, GPU memory capacity, memory bandwidth and compute limits.
- **Serving and scheduling:** request queues, continuous batching, chunked prefill, token budgets and the latency/throughput trade-off.
- **Performance measurement:** time to first token (TTFT), time per output token (TPOT), end-to-end latency, throughput, concurrency and tail latency.
- **Reliability and economics:** admission control, overload, cancellation, deadlines, stream failures, and cost per useful output token under a latency target.

Start with the request lifecycle, memory and performance metrics. Use the other areas to orient myself and select a follow-up; this is an optional refresher scope.

### Intended outcome

Draw the request path in my own words, make one rough memory estimate, and record the most important gaps and the next thread. Write my prediction and first explanation before using LLM hints or review.

### Optional reference

[Hugging Face: continuous batching architecture](https://huggingface.co/docs/transformers/main/continuous_batching_architecture) connects scheduling, prefill/decode and KV cache management.

## Question and prediction

## Work

## Findings

## Understanding

## Next thread
