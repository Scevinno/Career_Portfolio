#########################################################################
# Site-styled figure export for the RAG help-desk write-up
#########################################################################

# Reads rag_evidence.json (written by run_rag_evidence.py) and renders
# the retrieval-score figure used in the post, in the website's palette
# on a transparent background.
#
#   rag_retrieval_scores.png - relevance score of the three distinct
#                              chunks retrieved for each test question,
#                              against the retriever's 0.25 threshold
#
#   python scripts/render_rag_figures.py <dir holding rag_evidence.json>

import json
import os
import sys

import matplotlib.pyplot as plt

OUT_DIR = r"C:\Career_Portfolio\img\posts"

TEXT = "#9fb0c0"        # site --text-dim
BRIGHT = "#eef2f7"      # site --text
TEAL = "#17a589"        # site accent
VIOLET = "#a78bfa"      # secondary accent
GREY = "#5c6b7f"

plt.rcParams["font.family"] = "sans-serif"

data = json.load(open(os.path.join(sys.argv[1], "rag_evidence.json"), encoding="utf-8"))
questions = data["questions"]

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)

for row, rec in enumerate(reversed(questions)):
    # every chunk is stored twice, so keep the first score per chunk id
    seen, scores = set(), []
    for s in rec["scores"]:
        if s["id"] not in seen:
            seen.add(s["id"])
            scores.append(s["score"])
    colour = TEAL if rec["scope"] == "in" else VIOLET
    ax.plot([min(scores), max(scores)], [row, row], color=colour, alpha=0.35, lw=2)
    ax.scatter(scores[1:], [row] * len(scores[1:]), s=60, color=colour, alpha=0.55,
               edgecolor="none", zorder=3)
    ax.scatter(scores[0], row, s=120, color=colour, edgecolor=BRIGHT, lw=1.2, zorder=4)

ax.axvline(0.25, color=BRIGHT, lw=1.4, ls="--")
ax.text(0.255, len(questions) - 0.45, "retriever threshold 0.25", color=BRIGHT,
        fontsize=10.5, va="center")

ax.set_yticks(range(len(questions)))
ax.set_yticklabels([r["question"].replace("  ", " ") for r in reversed(questions)])
for label, rec in zip(ax.get_yticklabels(), reversed(questions)):
    label.set_color(TEAL if rec["scope"] == "in" else VIOLET)

ax.set_xlim(0, 0.85)
ax.set_ylim(-0.6, len(questions) - 0.1)
ax.set_xlabel("relevance score (1 − cosine distance)", color=TEXT, fontsize=11, labelpad=9)
ax.set_title("The three closest knowledge-base entries for each question",
             color=TEXT, fontsize=14, pad=16)

ax.tick_params(colors=TEXT, labelsize=10.5)
ax.tick_params(axis="y", length=0, pad=10)
ax.grid(axis="x", color=TEXT, alpha=0.12, lw=1)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GREY)

plt.tight_layout()
plt.savefig(OUT_DIR + r"\rag_retrieval_scores.png", transparent=True, bbox_inches="tight")
plt.close()
