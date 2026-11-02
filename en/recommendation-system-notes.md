---
layout:     post
lang:       en
permalink:  /en/recommendation-system-notes/
title:      "Some Observations on Recommender Systems"
subtitle:   "Traffic tiering, multiple stakeholders and problem definition"
zh_url:     /2023/04/23/recommendation-system-notes/
date:       2023-04-23
author:     Zhang Xian
tags:
    - Recommender systems
---

## Douyin

Douyin combines recommendation with product mechanics. It tiers traffic and promotes content along three dimensions: traffic pool × recommendation stage (cold start / learning / normal recommendation) × recommendation algorithm. The idea should carry over to any platform with a lot of new content, such as YouTube or an ad system.

Recommendation is a multi-stakeholder task and has to balance several parties, above all the user. Compared with advertising, it weighs ecosystem health and long-term value more (new users and new items, diversity). Advertising usually optimizes only the current eCPM, and its evaluation is narrower: everything points to money. Recommendation has more metrics, and therefore more room on the technical side.

Problem definition is the core difficulty in recommendation. It has to match the company's current stage and ecosystem goals: grow watch time, grow users, or widen the diversity of the user base. These are not purely technical questions; they are set by the market and the company's strategy. For example, with page-wise metrics, how do you trade off discovery time against familiar time (FM vs UF)?
