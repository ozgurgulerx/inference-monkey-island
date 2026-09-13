# Qwen3.8-27B

[Architecture notes](architecture.md) · [Experiments](experiments/) · [All models](../README.md)

[Day 001 journal](../../days/day-001-250912-leto-inference-refresher-a-new-hope/README.md)

## Overview
Qwen3.8 is a dense (not MoE) vision-language model built on Qwen3.6 architecture. 
Unique point with qwen3.8 is the hybrid attention. 
It has gated DeltaNet layers mixed with full attention layers.

1. Architecture and memory 
(DeltaNet + FFN → DeltaNet + FFN → DeltaNet + FFN → full attention + FFN) X 16 = 64 layers 
Giving 48 DeltaNet layers and 16 full-attention layers.
Full attention layers have 24 query heads, four KV heads, and a head dimension of 256.

Mweights​≈27×109×2=54 GB≈50.3 GiB

2. Baseline serving 
3. PRefill versus decode 
4. Hybrid Attention
5. Quantization and batching 
6. Reasoning and speculation


[HF Model Card](https://huggingface.co/Qwen/Qwen3.8-27B?utm_source=chatgpt.com)

[Start here — Qwen inference guide, from first session to deeper experiments](list01-study-levels.md)




## Model map

![Qwen3.8 (27B) architecture: gated DeltaNet and full-attention layers](qwen3.8-27b-model-map.webp)

Diagram credited to Sebastian Raschka in the image.
