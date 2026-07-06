---
layout: post
title: Clustering the S&P 500 by Market Behaviour
image: "/img/posts/sp500_clustering.svg"
tags: [Machine Learning, Clustering, Python]
summary: "K-means groups 499 S&P 500 companies by how they actually traded and earned through 2025 — built on a dataset I assembled from scratch — to test whether stocks really behave the way their sector labels say they should."
stack: "Python · pandas · scikit-learn · yfinance"
metrics:
  - value: "499"
    label: "companies clustered"
  - value: "6"
    label: "behavioural groups found"
  - value: "28%"
    label: "of tech stocks behave alike"
---

Every S&P 500 company carries a sector label — Utilities, Information Technology, Health Care — assigned by a classification committee, not by the market. I built a dataset of how each of 499 companies **actually behaved through 2025** — returns, volatility, valuation, profitability, growth — and let a K-means model group them with no knowledge of those labels. The question: do stocks behave like their sector says they should?

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Data Overview](#02-data-overview)
- [03. Data Preparation](#03-data-preparation)
- [04. Choosing How Many Clusters](#04-choosing-how-many-clusters)
- [05. Training the Model & Profiling the Clusters](#05-training-the-model--profiling-the-clusters)
- [06. Do Stocks Follow Their Sector Labels?](#06-do-stocks-follow-their-sector-labels)
- [07. Growth & Next Steps](#07-growth--next-steps)

---

## 00. Project Overview

**Context**

Sector labels drive real decisions — index funds, sector ETFs, portfolio diversification rules all lean on them. But a label is a statement about what a company *sells*, not how its stock *behaves*. If two "diversified" holdings actually move, earn and price identically, the diversification is on paper only. Clustering by measured behaviour puts the labels to the test.

**Actions**

There was no ready-made table for this, so I built one: for every current S&P 500 constituent I assembled ten 2025-anchored metrics — price behaviour computed from a full year of daily prices, fundamentals from the four quarterly statements ending within 2025 — into one row per company. On that table I ran an unsupervised **K-means** pipeline: min-max scaling, an elbow (WCSS) search for the right number of clusters, then profiling each cluster and mapping its sector make-up.

**Growth & Next Steps**

Heavy-tailed ratios like P/E compress under min-max scaling, so a log-transform or robust scaler is the next modelling step, followed by a silhouette check on the cluster count. The more ambitious extension is clustering on daily return correlation — grouping stocks by how they move *together*, not just by their summary statistics.

---

## 01. Results

The model settled on **six behavioural groups** — and they only partly respect the sector map. Three findings stand out:

**Some sectors are real behavioural tribes.** **71% of Utilities** and **67% of Consumer Staples** companies land in the same cluster — a low-beta, high-dividend, low-volatility group that behaves exactly like the defensive reputation of those sectors. Their label genuinely tells you how they trade.

**Tech is not one thing.** Information Technology is the most fragmented sector in the index: its **biggest cluster holds only 28%** of its companies, with the rest spread across four other groups. Two stocks with the same "tech" label can sit in completely different behavioural worlds — one trading like a steady industrial, another like a rocket.

**The market has a momentum tribe that ignores sector lines.** A small cluster of 22 companies — Palantir, Robinhood, Micron, Carvana, GE Vernova among them — averaged an **+86% return in 2025** at nearly double the market's volatility and a beta of 2.0. It spans tech, financials, industrials and consumer names: behaviour, not sector, is what these stocks share.

| Cluster | Companies | Return | Div. yield | Volatility | Beta | Character |
|---|---|---|---|---|---|---|
| Defensive income | 69 | +5% | 3.8% | 24% | 0.3 | Utilities & staples heartland |
| Quality value | 51 | +13% | 3.3% | 26% | 0.7 | Cheap (P/E ~20), high-margin payers |
| Steady mega-caps | 145 | +8% | 1.0% | 27% | 0.7 | The index's calm core |
| Cyclical traders | 55 | +14% | 2.6% | 37% | 1.2 | Energy & materials tilt |
| Volatile growth | 80 | +5% | 0.4% | 44% | 1.4 | High-beta, growth-priced |
| Momentum high-flyers | 22 | +86% | 0.4% | 62% | 2.0 | The 2025 rocket ships |

---

## 02. Data Overview

The dataset is the part of this project I'm proudest of — every number in it is computed from dated source data, anchored to **calendar 2025** so no metric leaks information from a different period. Price metrics come from 2025's ~250 trading days (with the last 2024 close as baseline); fundamentals are trailing-twelve-month figures summed from the four quarterly statements ending within 2025, falling back to the nearest full fiscal year for off-cycle reporters. Four constituents without full 2025 history (2026 additions and spin-offs) were dropped, leaving **499 companies**.

| Variable Name | Variable Type | Description |
|---|---|---|
| ticker / company | Identifier | Ticker symbol and company name — dropped before modelling |
| sector | Label (held out) | One of the 11 GICS sectors — never shown to the model, used only to evaluate the clusters afterwards |
| annual_return_pct | Behavioural | Compounded 2025 price return |
| dividend_yield_pct | Behavioural | 2025 dividends paid ÷ start-of-year price |
| volatility_pct | Behavioural | Annualised standard deviation of daily returns |
| beta | Behavioural | Sensitivity to S&P 500 index moves across 2025 |
| pe_ratio | Valuation | Dec-2025 close ÷ trailing diluted EPS (blank if EPS ≤ 0) |
| roe_pct | Profitability | Return on equity, trailing twelve months |
| profit_margin_pct | Profitability | Net margin, trailing twelve months |
| eps_growth_pct | Growth | EPS growth, 2025 vs 2024 |
| revenue_growth_pct | Growth | Revenue growth, 2025 vs 2024 |
| market_cap_b | Scale | Market capitalisation in $ billions |

One deliberate choice: **missing values stay missing**. A company with negative earnings has no meaningful P/E — writing one in would be inventing data. Gaps are handled at the modelling stage instead, where dropping them is an explicit, visible decision.

---

## 03. Data Preparation

K-means has no concept of a label, an identifier, or a text column — it just measures distances between rows. So preparation means three things: deal with the gaps, remove everything that isn't a behavioural measurement, and put every metric on the same scale.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

data_for_clustering = pd.read_csv("sp500_2025_metrics.csv")

# rows with any missing metric are dropped — 499 down to 422
data_for_clustering.isna().sum()
data_for_clustering.dropna(how="any", inplace=True)

# identifiers and the sector label are held out of the feature set
data_for_clustering_scaled = data_for_clustering.drop(["ticker", "company", "fundamentals_basis", "sector"], axis=1)

# normalise every metric to the same 0-1 footing
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering_scaled),
                                          columns=data_for_clustering_scaled.columns)
```

The scaling step is not optional for K-means. Market cap runs into the thousands of billions while beta lives between 0 and 3 — unscaled, every distance the model measures would be a market-cap comparison with noise attached. Min-max normalisation gives each of the ten metrics an equal vote.

The **sector column is deliberately held out**. The whole point is to see what groups emerge *without* it — it comes back at the end as the answer key.

---

## 04. Choosing How Many Clusters

K-means needs to be told how many groups to find. The standard tool is the **elbow method**: fit the model at every k from 1 to 11, record the Within-Cluster Sum of Squares (how tightly each cluster hugs its centre), and look for the point where adding another cluster stops buying much tightness.

```python
k_values = list(range(1, 12))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)

plt.plot(k_values, wcss_list)
plt.title("Cluster Sum of Squares")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()
```

The curve bends gradually rather than snapping at one obvious point — common on real financial data, where group boundaries are soft. I chose **k = 6**: past six, each extra cluster shaved little off the WCSS, and the six groups that emerge are distinct enough to describe in plain English — which is its own kind of validation for an unsupervised model.

---

## 05. Training the Model & Profiling the Clusters

```python
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(data_for_clustering_scaled)

# attach each company's cluster back onto the readable data
data_for_clustering["cluster"] = kmeans.labels_
data_for_clustering["cluster"].value_counts()
```

The labels go back onto the **unscaled** table, which matters for the next step: profiling in real units. Averaging each metric per cluster turns six anonymous group numbers into six recognisable investor archetypes — the table in [01. Results](#01-results):

```python
cluster_summary = data_for_clustering.groupby("cluster")[["annual_return_pct", "dividend_yield_pct",
    "volatility_pct", "beta", "pe_ratio", "roe_pct", "profit_margin_pct",
    "eps_growth_pct", "revenue_growth_pct", "market_cap_b"]].mean().reset_index()
```

Reading the profiles is where the model earns its keep. A 3.8% average yield with a 0.3 beta is unmistakably a defensive income group; a 62% volatility with a 2.0 beta and +86% returns is unmistakably the momentum crowd. The model was never told these categories exist — it found them in the numbers.

---

## 06. Do Stocks Follow Their Sector Labels?

Now the held-out sector column comes back. Cross-tabulating sector against cluster — normalised within each sector — shows exactly how much of each sector shares one behaviour:

```python
sector_mix = pd.crosstab(data_for_clustering["sector"], data_for_clustering["cluster"], normalize="index")

ax = sector_mix.plot(kind="barh", stacked=True, figsize=(12, 6), width=0.85,
                     edgecolor="white", linewidth=1)
for bars in ax.containers:
    ax.bar_label(bars, labels=[f"{v:.0%}" if v >= 0.08 else "" for v in bars.datavalues],
                 label_type="center", fontsize=10, color="white")

plt.title("Cluster Composition Within Each Sector")
plt.xlabel("Share of sector's companies")
plt.ylabel("")
plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
plt.box(False)
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
```

Each sector's bar splits into the behavioural clusters its companies actually landed in — and the spectrum runs from tribal to scattered:

| Sector | Largest single cluster | Reading |
|---|---|---|
| Utilities | 71% | The label works — one shared behaviour |
| Consumer Staples | 67% | Same — the classic defensive block |
| Energy | 60% | Cohesive, in the cyclical-trader group |
| Health Care | 56% | Mostly one group, with a volatile fringe |
| Financials | 39% | Split three ways — banks ≠ exchanges ≠ insurers |
| Information Technology | 28% | The label tells you almost nothing about behaviour |

For a portfolio builder the practical takeaway is direct: holding five different tech names may still be one concentrated behavioural bet — or five genuinely different ones. The sector label can't tell you; the behaviour can.

---

## 07. Growth & Next Steps

Concrete improvements queued for the next iteration:

- **Tame the heavy tails.** A handful of true-but-extreme values (a P/E above 6,000, an ROE near 4,000%) compress everyone else into a sliver of the min-max scale, muting those metrics' influence. Log-transforming the ratio columns — or switching to a robust scaler — would let them speak properly.
- **Validate the cluster count.** The elbow read was a judgement call; a silhouette-score sweep across k would put a number on it.
- **Cluster on co-movement.** Summary statistics describe each stock alone. Clustering on the correlation of daily returns would group stocks by how they move *together* — the definition of behaviour that matters most for diversification.

---
