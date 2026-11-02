---
layout: page
lang: en
title: About
subtitle: AI algorithm expert with fifteen years in search, advertising and recommendation, now working on LLM agents and real-time voice agents.
permalink: /about/
description: Zhang Xian's background, core skills and experience.
---

I am Zhang Xian, an algorithm engineer based in Beijing. I spent my first decade or so on search, advertising and recommendation. Since September 2025 I have worked on LLM agents, mainly full-duplex real-time voice agents. [中文版](/zh/about/)

The two phases look far apart, but the problem is the same: under strict latency and cost constraints, pick the right option from a huge candidate set, and keep the whole system measurable and continuously improvable.

## Core skills

### Real-time voice agents

- Design and implementation of full-duplex voice dialogue engines for outbound sales and customer service
- Semantic interruption, and detection and handling of overlapping speech
- SOP-driven flow orchestration: compiling business processes into executable dialogue flows
- End-to-end latency optimization, prompt-cache hit rate, and tail latency under concurrency

### LLM decision and generation

- A two-stage architecture of rule retrieval plus an LLM selector: choose the flow node and applicable rules first, then generate the reply
- Turning selection into yes/no judgments answered in batch by a lightweight decision model, for speed and calibratability
- Substantially lower token use and latency than general agent frameworks at comparable quality

### Inference and systems

- GPU inference capacity planning and profiling (H20, H200): concurrency, P95 and prefill contention
- The low-level mechanics of the inference stack: tensor cores and low-precision formats, FlashAttention and MLA, MoE communication and compute (see [From SM100 to SM90: a layer-by-layer walkthrough of the DeepSeek-V4.1 inference stack]({{ '/2026/10/28/deepseek-v41-sm90-inference-walkthrough/' | relative_url }}), written in Chinese)

### Evaluation and iteration

- Design of an SOP-compliance evaluation system
- Migrating evaluation infrastructure to Langfuse
- Turning production errors into correction samples: retrieved similar cases are injected into the prompt and also kept as later training data

### Search, advertising and recommendation

- Fifteen years of ranking, CTR prediction and recommender-system work, in both algorithms and engineering
- Attention to problem definition itself: trade-offs among stakeholders, new users and new content, diversity and long-term ecosystem health

## Experience

| Period | Area | Work |
|---|---|---|
| Sep 2025 to present | LLM agents | AI algorithm expert, real-time voice agents |
| Previous decade or more | Search, advertising and recommendation | Algorithm work at several internet and technology companies |

## Currently interested in

- Reinforcement learning and post-training for dialogue: self-play, GRPO, DPO, and potential-based reward design
- Deal-closing agents that can converse with themselves and keep improving
- Adapting LLM inference to domestic and existing hardware

## Contact

- Email: [{{ site.email }}](mailto:{{ site.email }})
- GitHub: [github.com/{{ site.github_username }}](https://github.com/{{ site.github_username }})
