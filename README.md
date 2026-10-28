# lxsabrina.github.io

张弦的个人网站与技术博客，基于 GitHub Pages 与 Jekyll。

- 首页：`index.html`
- 关于：`about.md`
- 文章列表：`writing.html`
- 文章：`_posts/`，文件名格式 `YYYY-MM-DD-slug.md`
- 文章配图：`img/posts/<文章名>/`
- 样式：`assets/css/site.css`

新文章的 front matter：

```yaml
---
title: 标题
subtitle: 副标题
date: 2026-10-28
tags:
    - 标签
---
```

本地预览：

```bash
gem install github-pages
jekyll serve
```
