---
title: 推荐系统的几点观察
subtitle: 流量分层、多利益方与问题定义
title_en: Some Observations on Recommender Systems
subtitle_en: Traffic tiering, multiple stakeholders and problem definition
date: 2023-04-23
tags:
    - 推荐系统
---

## 抖音

抖音推荐+机制玩法。抖音是将流量分层，流量池 × 推荐阶段（冷启/学习/正常推荐）× 推荐算法，三个维度精细化推广，推而广之，对于新内容多的平台应该都不错，比如youtube,广告系统.

推荐算是个 multi-stakeholder 的任务，还是要兼顾多方利益，尤其是用户侧. 相比广告差异，更加考虑生态和长期利益(新用户新物品,多样性），广告一般考虑只当前ecpm最优，评估指标也更多样,  而不是像广告一切向“钱”看, 所以在技术上也有更多空间

推荐系统中问题定义是核心难点，问题定义需要结合企业当前阶段和生态目标，是要涨时长，还是涨用户，还是扩大用户群体多样性.  这些不是单独技术问题，还有当前市场和企业发展战略决定。比如: Page-wise 指标吧，fm v.s. uf 如何权衡 discovery time 还有 familiar time
