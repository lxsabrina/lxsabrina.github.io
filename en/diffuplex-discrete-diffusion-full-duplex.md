---
layout:     post
lang:       en
permalink:  /en/diffuplex-discrete-diffusion-full-duplex/
title:      "Reading Diffuplex: Full-Duplex Spoken Language Modeling with Discrete Diffusion"
subtitle:   "Autoregression decides when to speak, parallel diffusion decides what to say"
zh_url:     /2026/11/04/diffuplex-discrete-diffusion-full-duplex/
date:       2026-11-04 12:30:00 +0800
author:     Zhang Xian
tags:
    - Voice agents
    - Full duplex
    - Diffusion language models
    - Paper notes
---

## Summary

Full-duplex spoken models handle turn-taking well but trail autoregressive (AR) spoken models on intelligence. Diffuplex (NVIDIA, arXiv 2610.17780) explains this as follows: existing models squeeze "when to speak" and "what to say" onto one AR timeline, so the training objectives conflict, and semantic computation is tied to a fixed perception frame rate (the paper cites Moshi's 80 ms), leaving no room for extra reasoning.

Its answer is to separate the two jobs **at training time**. An AR mode tracks the user's speech every 80 ms and emits an interaction state; a diffusion mode generates the reply content in parallel on a 256-token canvas. Both modes share one backbone and keep separate KV caches.

This note relies only on the paper. All numbers are author-reported and not reproduced. The paper says code and checkpoints will be released "soon"; they were not available at the time of writing.

![Figure 1: Diffuplex inference structure](/img/posts/diffuplex/01-architecture-en.svg)

*Figure 1: the AR mode handles state, the diffusion mode handles content, and TTS can start once the prefix is ready.*

## Key conclusions

- The core design is a separation of duties: the AR mode only predicts four interaction states, content generation goes to discrete diffusion, and the two are trained apart.
- Efficiency: about 13.0 tokens finalized per forward pass (AR baselines: 1.00); in thinking mode about 1,872 tokens/s, versus 394 and 300 for two AR baselines in the table.
- Intelligence: the full-duplex system scores 82.3 on VoiceBench, 24.2 points above the strongest full-duplex AR baseline in the table (58.1); the non-duplex Stage 2 model scores 84.1 (86.9 with thinking).
- Conversational behavior: interruption latency is 0.239 s (the variant trained with Fisher data), close to Moshi's 0.257 s.
- The costs are explicit: going full duplex loses 1.8 points relative to Stage 2, and diffusion decoding is slightly below AR decoding (84.1 vs 85.1).
- The training recipe matters a lot: removing the text anchor drops the score to 66.1, and removing Stage 1 drops it to 29.9.

## 1. The problem

**Conclusion: the authors attribute the lack of intelligence in full-duplex models to structure, not to data volume.**

On a single AR timeline, the model must decide at each step both whether to speak now and what to say. The paper points to two consequences:

1. Conflicting objectives: timing decisions and content generation pull on the same loss.
2. Compute tied to the frame rate: semantic computation is bounded by the perception frame rate (one step per 80 ms), so the model cannot spend extra compute thinking before it replies.

## 2. Architecture

**Conclusion: a text discrete-diffusion language model as the backbone, plus a light state head, gives two working modes.**

| Component | Detail |
|---|---|
| Speech front end | Parakeet speech encoder, connected to the backbone's embedding space through a randomly initialized projector |
| Backbone | DiffusionGemma (26B-A4B MoE, a text discrete-diffusion language model) |
| AR mode | Causal attention; every 80 ms it predicts one of `<listen>`, `<speak>`, `<bos>`, `<eos>`; the only additions are a state embedding and a state head |
| Diffusion mode | Works on a 256-token "canvas", conditioned on speech features and the prompt, and denoises the reply in parallel |
| Speech synthesis | CosyVoice 3; the paper says the design is modular and any TTS can be swapped in |

The two modes share parameters but keep their own KV caches.

## 3. Training in three stages

**Conclusion: align the modality first, then learn instruction following, and add full duplex last; each step has a matching ablation.**

| Stage | What it does | Notes |
|---|---|---|
| Stage 1: modality alignment | Warm up the projector on LibriSpeech (about 960 h), then train projector and encoder jointly on an English ASR mixture | The diffusion LM stays frozen; WER 2.3 (test-clean), 4.2 (test-other) |
| Stage 2: spoken instruction following | About 600K speech-instruction examples plus 5% ASR replay | About 70% of responses are direct answers without thinking; a frozen copy of the original diffusion model acts as teacher for "text-anchored distillation" (written prompt in, forward-KL matching on diffusion and AR outputs) |
| Stage 3: full-duplex interaction | Freeze the response generator; train a four-state interaction head and a listening LoRA active only on the speech pathway | About 120 h of clean synthetic conversations and 70 h of internal conversational data; a variant adds the Fisher telephone corpus; noise augmentation is used |

## 4. Inference

**Conclusion: prefix anchoring lets TTS start before the canvas is finished.**

1. The AR policy consumes streaming audio and emits states.
2. On `<bos>`, accumulated speech features and the prompt are prefilled into the diffusion model to build the cache.
3. Diffusion denoises a 256-token canvas with an entropy-bounded sampler (β = 0.1, up to 48 passes), stopping early on convergence.
4. Prefix anchoring: after at most K passes, the first n = 8 tokens are frozen so TTS can start (K = 4 in the reported setting).
5. Modified schedule: temperature anneals from 0.7 to 0.2, the convergence threshold γ is relaxed to 0.025, and degenerate loops are cut.
6. On `<eos>`, playback is truncated; if the user interrupts within one second, the unspoken part of the canvas is dropped from the context.

K sets the latency-quality trade-off (RTX 6000, post-trigger latency, excluding the AR turn-decision time):

| K | Latency | VoiceBench overall |
|---|---|---|
| 2 | 251 ms | 80.5 |
| 4 | 296 ms | 82.3 |
| 6 | 309 ms | 82.6 |
| Uncapped | 497 ms | 84.0 |

![Figure 2: latency vs quality across K](/img/posts/diffuplex/02-k-tradeoff-en.svg)

*Figure 2: from K=2 to K=4, latency rises 45 ms and the score rises 1.8; from K=4 to K=6 the gain is only 0.3.*

## 5. Results

**Conclusion: the gains in intelligence and speed are large, but the comparisons are not fully like-for-like.**

### 5.1 Intelligence and efficiency

- VoiceBench overall: 86.9 with thinking, 84.1 without (Stage 2), and 82.3 for the full-duplex Stage 3 system.
- Efficiency: 13.0 TPF versus 1.00 for AR baselines and 0.37 for DIFFA-2; throughput of about 1,872 tokens/s versus 394 for Nemotron-3 Omni and 300 for Qwen3-Omni-Thinking. Diffuplex has 4B active parameters against 3B for those two.
- The same recipe on Nemotron-Labs-Diffusion-8B reaches about 78.7 with thinking.

### 5.2 Conversational behavior

| Metric | Result |
|---|---|
| Turn-taking rate (TOR, v1) | 0.933 (with Fisher), 1.000 (without) |
| Interruption TOR | 0.945, 1.000 |
| Interruption latency | 0.239 s (with Fisher); Moshi 0.257 s |
| Synthetic pause scenario (lower is better) | 0.380 without Fisher, 0.679 with Fisher |

Fisher data makes turn-taking faster but pause handling worse: a trade-off between latency and robustness to hesitation.

### 5.3 Ablations (VoiceBench, no thinking)

| Setting | Overall |
|---|---|
| Diffuplex (spoken input) | 84.1 |
| Text input (upper bound) | 90.1 |
| AR decoding instead of diffusion | 85.1 |
| Without text anchor | 66.1 |
| Without Stage 1 | 29.9 |
| Frozen diffusion LM in Stage 2 | 64.7 |

The authors conclude that Stage 1 alignment and the text anchor are both essential, and that controlled adaptation of the backbone is needed. Note that AR decoding scores a point higher than diffusion: what diffusion buys is speed, not a higher single-step quality.

## 6. Limitations and items to verify

All of the following come from the paper's own text or tables:

- Stage 3 loses 1.8 points relative to Stage 2. The authors say this fits streaming decisions and early-prefix decoding but do not isolate the cause.
- The K-sweep latency excludes the AR turn-decision time and was measured on an RTX 6000, while throughput was measured on a GB300. The GB300 deployment uses FP8/NVFP4 quantization and other GPUs use BF16, so efficiency comparisons are not perfectly like-for-like.
- TPF counts answer-text tokens and is not directly comparable to speech-token or codec-level multi-token-prediction methods; the paper includes no direct MTP baseline.
- The second diffusion backbone scores lower, which the authors attribute to weaker text capability, so results depend on the backbone.
- Closed-source real-time models (e.g. GPT-Realtime) are excluded because API overhead would distort latency.
- The paper has no dedicated limitations section.

**My inference (not a conclusion of the paper):** the design turns content generation into one parallel computation that can spend a few extra iterations, so the price of "thinking" is a one-off latency rather than a cost paid at every 80 ms step. This agrees with the K sweep above, but the paper does not phrase it this way.

## Conclusion

Diffuplex is interesting for its structure: a light AR policy for timing, discrete diffusion for parallel content, and prefix anchoring to join them. By the authors' numbers this closes much of the VoiceBench gap between full-duplex and AR models while keeping interruption latency close to Moshi's. The comparison basis, the quantization settings and the dependence on the backbone are all things to keep in mind when reading the numbers, and nothing can be reproduced until the code and checkpoints are out.

## References

- [Full-duplex Spoken Language Modeling with Discrete Diffusion (arXiv 2610.17780)](https://arxiv.org/abs/2610.17780)
- [HTML version of the same paper](https://arxiv.org/html/2610.17780v1)

Related reading: [Reading Full-Duplex Voice Models: MiniCPM-o 4.5 and Realtime-Venus]({{ '/en/minicpm-o-45-realtime-venus/' | relative_url }}).

*Note: all numbers are author-reported and were not independently reproduced.*
