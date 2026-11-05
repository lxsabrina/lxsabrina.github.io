---
layout:     post
lang:       en
permalink:  /en/barge-in-benchmarks-2026/
title:      "How to Evaluate Barge-In: SID-Bench, EchoChain, IHBench and Instruct-FD"
subtitle:   "From stopping correctly, to recovering after the stop, to following turn-taking instructions"
zh_url:     /2026/11/05/barge-in-benchmarks-2026/
date:       2026-11-05 12:00:00 +0800
author:     Zhang Xian
tags:
    - Voice agents
    - Full duplex
    - Evaluation
    - Paper notes
---

## Summary

Barge-in is when the user speaks while the system is still talking. Early evaluations asked one question: did the system stop, and how fast? Several 2026 benchmarks split the problem further: **SID-Bench** (Qwen team) tests whether the system should stop; **EchoChain** (Labelbox) and **IHBench** (Boson AI) test whether what it says afterwards is right; **Instruct-FD** (Boson AI) tests whether it can change its turn-taking behaviour on instruction.

This note relies only on those four papers (for Instruct-FD, only the abstract page). My reading is that the four benchmarks measure three different layers of interruption handling, so their numbers cannot be compared with each other. For anyone building a voice agent, the second layer deserves the most attention, because state updates after the stop are where users feel the awkwardness.

## Key conclusions

- Interruption handling has at least three layers: detect and stop, recover and update state, follow policy. Each benchmark covers one; none covers all.
- Measuring reaction time alone misses most of the problem. In EchoChain no model passes more than 50% of conversations in full (best 48.5%); in IHBench, continuing a cut-off half-utterance is correct only 7% to 31% of the time for the GPT family.
- Both "recovery" benchmarks rely on LLM judging, but validate it differently: IHBench compares against human annotators; EchoChain uses human-applied rubrics.
- SID-Bench's APT turns false stops and missed stops into one unit, seconds. It is convenient, but the penalty rules are heuristic and the paper leaves correlation with human judgement to future work.
- All results are author-reported, on different targets and protocols. Read them as trends.

## A few concepts first

- **Barge-in**: the user starts speaking while the system is talking. The system must decide whether it is a real interruption or a backchannel ("uh-huh"), a cough, or background noise.
- **VAD** (voice activity detection): looks only at whether there is sound, so it also fires on backchannels and noise. SID-Bench calls such systems "trigger-happy".
- **False stop / missed stop**: stopping when it should have continued (FP), or continuing when it should have stopped (FN). FP wastes a turn; FN makes the user wait through audio they no longer want.
- **State update**: an interruption often carries new information ("wait, make it Friday"). The system has to stop and also reflect that information afterwards.
- **Workflow state machine**: in multi-step tasks (verification, booking) the system is at some step, with some steps done. After an interruption it should return to the right step, not restart or skip.

## 1. The four benchmarks at a glance

![Figure 1: Three layers of one interruption](/img/posts/barge-in-benchmarks/01-three-layers-en.svg)

*Figure 1: an illustration. The four benchmarks grouped by what they measure; the grouping is my own.*

| Benchmark | Team | Measures | Size | Scoring |
|---|---|---|---|---|
| SID-Bench | Qwen team | Semantic interruption detection | 3,700 clips (about 10 h) | FIR, IRL, APT, computed |
| EchoChain | Labelbox | State-update reasoning after interruption | 200 interrupted conversations | Human rubrics, pass rates |
| IHBench | Boson AI | Workflow recovery after interruption | 45 conversations, 428 interruption points | LLM judge, checked against humans |
| Instruct-FD | Boson AI | Instructed turn-taking control | not given on the abstract page | LLM judge |

## 2. SID-Bench: should it stop?

**Verdict**: it makes "stopping correctly" computable and ships a small reference model.

- **Data**: Chinese and English recordings from a professional data vendor plus an English subset of Switchboard. Per language: 500 "interruption at the start", 600 "interruption in the middle", 500 "uninterrupted"; plus 200 environmental-noise and 300 silence clips. Total: 2×(500+600+500)+500 = 3,700.
- **Labels**: an LLM inserts a break point at the first substantive word; Kaldi forced alignment gives the start of that word's first phoneme as the ground-truth interruption time.
- **Metrics**: FIR (false interruption rate), IRL (interruption response latency), APT (average penalty time). APT is explained in Figure 2.

![Figure 2: How APT penalises each outcome](/img/posts/barge-in-benchmarks/02-apt-penalty-en.svg)

*Figure 2: summarised from the paper's textual description; unit is seconds.*

Average results from the paper's Table II (author-reported):

| System | Avg IRL (s) | Avg FIR | Avg APT (s) |
|---|---|---|---|
| FSMN-VAD | 0 | 0.840 | 3.627 |
| Freeze-Omni (Silero VAD) | 0.173 | 0.532 | 2.129 |
| FireRedChat (pVAD) | 1.045 | 0.476 | 2.627 |
| Moshi | 2.517 | 0.118 | 3.192 |
| SID-model | 0.389 | 0.124 | 0.711 |

Reading the table: plain VAD reacts fastest but stops on a large share of non-interruptions; Moshi rarely stops falsely but stops slowly. The SID-model feeds an audio encoder into a Qwen3-0.6B LLM that makes an "interrupt / continue" decision, and stops only after three consecutive "interrupt" predictions. It pays about 0.39 s of IRL, but its APT is about a third of the best baseline's.

**Caveats** (stated by the paper or implied by its description):

- Moshi was tested by playing audio at a fixed time during a seeded response, while VAD baselines were scored by onset times, so the protocols are not fully equivalent. Moshi has no Chinese result.
- Labels depend on LLM-generated break points not checked against human annotation; silence clips are synthetic concatenations.
- APT's penalties (e.g. a false stop costs the whole turn) are heuristic; correlation with human judgement is future work.
- Single run, no variance; the SID-model's fine-tuning data size is not given.

## 3. EchoChain: did the update actually happen?

**Verdict**: models mostly hear the interruption but do not use the new information correctly.

- **Design**: a planner model and a user-simulator model generate conversations. User speech is pre-synthesised with cloned voices, so every model hears identical audio. The first exchange is uninterrupted; each later barge-in is injected at a fixed offset from detected assistant speech onset, so timing is the same across models.
- **Metrics**: MPR (share of conversations where every rubric criterion passes) and MCP (share of individual criteria passed).
- **Failure taxonomy**: contextual inertia (acknowledges but content unchanged), interruption amnesia (applies the update, then drops it), objective displacement (answers only the interruption and loses the original task), other.

![Figure 3: EchoChain failure types](/img/posts/barge-in-benchmarks/03-echochain-failures-en.svg)

*Figure 3: redrawn from the paper's table; author-reported numbers, shorter is better.*

| Model | Contextual inertia | Interruption amnesia | Objective displacement | Other | Total failed | MPR (%) | MCP (%) |
|---|---|---|---|---|---|---|---|
| Amazon Nova Sonic 2 | 73 | 53 | 16 | 5 | 147 | 26.5 | 78.1 |
| Gemini Live-2.5-flash-native-audio | 38 | 65 | 64 | 0 | 167 | 16.5 | 64.8 |
| GPT-realtime-2025-08-28 | 29 | 45 | 35 | 1 | 110 | 45.0 | 81.1 |
| Grok Voice Agent | 42 | 39 | 16 | 6 | 103 | 48.5 | 85.5 |

I checked the table by hand: in every row the four categories sum to "Total failed", and (200 − Total failed)/200 equals MPR exactly (Nova: 53/200 = 26.5%). So "Total failed" appears to be the number of failed conversations, each assigned to one category. This is my inference from the table; the text does not spell it out.

One design point worth copying: in 48 sampled conversations, the interrupted versions had 92 failures against 55 for the same content without interruption, about 40% more failures attributable to the interruption ((92−55)/92 ≈ 40.2%). This separates task difficulty from interruption difficulty.

**Limitations** (stated by the paper): only task-relevant interruptions, no backchannels, side speech or noise; fixed offsets may not match natural interruption points; pass/fail rubrics hide severity; only four closed-source models; timing and prosody are not measured.

## 4. IHBench: back to the right step?

**Verdict**: the same "recovery" question, set inside a multi-step workflow, with a measurement of how it degrades as the conversation gets longer.

- **Data**: 45 synthetic conversations over 10 enterprise domains, 428 interruption points, 30.1 messages per conversation on average. Interruption types by share: pushback 24.5%, impatient 19.6%, normal 18.9%, topic switch 17.1%, filler 14.0%, correction 5.8%.
- **Metrics**: task fulfilment (TF) is a comparative LLM-judge win rate between the model's response and a baseline's, with 0.5 meaning parity; recovery quality (RQ) passes only if all 2 to 4 type-specific criteria are met.
- **Judge validation**: a second judge gives Spearman ρ of 0.99 (TF) and 0.95 (RQ); judge–human κ is 0.41 to 0.51, comparable to human–human κ (0.43, 0.40); model rankings agree with humans at ρ = 1.0 on RQ and 0.90 on TF.
- **Configurations**: 27 in total, 17 closed-weight and 10 open-weight.

Main findings (author-reported):

- **Filler is hardest**: after a cut-off "um…", GPT-family models continue the interrupted utterance correctly only 7% to 31% of the time; Gemini 2.5 manages 62% to 68%, while Gemini 3.x falls to 13% to 32%.
- **Later is worse**: TF drops 0.030 per turn on average; open-weight models decline about 3.3× faster than closed-weight ones (−0.053 vs −0.016).
- **Audio never beats text**: some open-weight models score about 8 TF points higher with text input.
- **TF and RQ only partly correlate**: the best on task fulfilment (GPT Realtime 2, 0.728) is not the best on recovery quality (Gemini 2.5 Flash with thinking, 0.704).

**Limitations** (stated by the paper): synthetic, English-only, 10 domains; rubrics inherit generator and judge bias; only response text is judged, not prosody.

## 5. Instruct-FD: whose rules?

I read only the abstract page. The question: applications want different turn-taking policies (proactive tutoring vs passive listening); can a full-duplex system change behaviour on instruction? The abstract says that among six full-duplex systems the best reaches 64.4% adherence, and proactive behaviours, especially the model's own backchanneling and interruption, are hardest. Dataset size, the judge model and per-system results are not on that page, so I do not go further.

## 6. The engineering view

**Where each layer fits**

| Layer | Fits | Precondition |
|---|---|---|
| Detect and stop (SID-Bench) | Choosing or training an interruption detector | Validate on your own audio and language; data covers Chinese and English only |
| Recovery (EchoChain, IHBench) | Evaluating end-to-end speech models or voice agents | Sample-check the judge yourself; synthetic data differs from real users |
| Policy (Instruct-FD) | Products needing different turn-taking styles | Details need the full paper |

**Comparison caveats**

- The "scores" mean different things: seconds, pass rates, win rates. Do not put them on one leaderboard.
- The targets differ: SID-Bench tests a detection module, EchoChain closed-source end-to-end models, and IHBench includes a text-input control.
- Judge bias: both recovery benchmarks use LLM judging; IHBench gives human-agreement numbers, EchoChain relies mainly on human-applied rubrics.

**Ideas to borrow directly**

1. Convert false stops and missed stops to one unit (the APT idea) and look at distributions, not means.
2. Inject the barge-in at "N seconds after assistant onset" so systems are comparable (EchoChain).
3. Run paired interrupted / uninterrupted controls to separate task difficulty from interruption effect.
4. Break results down by interruption type, and list "cut-off half-utterance" separately (IHBench finds it the most fragile).
5. Plot a depth curve over conversation turns instead of looking only at the first interruption.

## Conclusion

Barge-in is not one metric but a chain: stop correctly, say the right thing next, and follow the business rules. The 2026 benchmarks fill in the last two links, but each is small (hundreds of samples), mostly synthetic speech, and largely English. They work as a checkup, not as a leaderboard.

## References

- [SID-Bench: Semantic-Aware Interruption Detection in Spoken Dialogue Systems: Benchmark, Metric, and Model (arXiv 2603.24144)](https://arxiv.org/abs/2603.24144)
- [EchoChain: A Full-Duplex Benchmark for State-Update Reasoning Under Interruptions (arXiv 2604.16456)](https://arxiv.org/abs/2604.16456)
- [IHBench: Evaluating Post-Interruption Recovery in Voice Agents with Structured Workflows (arXiv 2606.19595)](https://arxiv.org/abs/2606.19595)
- [Instruct-FD: Can Your Full-Duplex Speech System Follow Turn-Taking Instructions? (arXiv 2607.20460)](https://arxiv.org/abs/2607.20460)

Related reading: [Seven Full-Duplex Speech Systems]({{ '/en/full-duplex-listening-time/' | relative_url }}), and [Reading Diffuplex]({{ '/en/diffuplex-discrete-diffusion-full-duplex/' | relative_url }}).

*Note: all numbers are author-reported and were not independently reproduced; the reading of "Total failed" and the 40% figure are my own checks of the tables.*
