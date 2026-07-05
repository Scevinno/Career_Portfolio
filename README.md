# Career Portfolio

Data portfolio of **Nikolas Scevko** — a hand-built dark "data-editorial" theme (no remote theme, no framework).

Live at: https://scevinno.github.io/Career_Portfolio

## Adding a project

1. Drop a markdown file in `_posts/` named `YYYY-MM-DD-slug.md`.
2. Front matter drives the case-study card on the home page:

```yaml
---
layout: post
title: Project Title
image: "/img/posts/cover.svg"        # cover image, lives in img/posts/
tags: [Machine Learning, Python]
summary: "One-to-two sentence hook shown on the card and page header."
stack: "Python · pandas · scikit-learn"
metrics:                              # up to ~3 headline numbers
  - value: "0.76"
    label: "R² on unseen sales"
---
```

3. Write the body in plain markdown — code blocks, tables and blockquote callouts are styled automatically.

## Toolkit section (measured library mix)

The "Python library mix" bars on the home page are computed, not guessed. `scripts/measure_libraries.py` attributes every executable line of the published projects' source code (blanks and comments excluded) to the library it calls, via import aliases and one-hop variable provenance:

```
python scripts/measure_libraries.py path/to/Model_Reg.py path/to/Model_For.py
```

When a new project is published, re-run it over all published projects' source files and update the percentages in `_data/toolkit.yml` (also bump `measured_lines` and `updated`).

## Stack

- Jekyll (GitHub Pages native build), kramdown + rouge
- Custom layouts in `_layouts/`, one stylesheet (`assets/css/style.css`), one small JS file (`assets/js/main.js`)
- Fonts: Space Grotesk (display), Inter (body), JetBrains Mono (data labels & code)
