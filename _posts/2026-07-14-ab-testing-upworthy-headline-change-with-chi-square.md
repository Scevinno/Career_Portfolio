---
layout: post
title: A/B Testing Upworthy Headline Change with Chi-Square
image: "/img/posts/upworthy_ab_test.svg"
tags: [A/B Testing, Hypothesis Testing, Python]
summary: "Upworthy ran the same headline twice — one word different. A chi-square test on ~24,800 randomised readers says the word genuinely changed how many people clicked."
stack: "Python · pandas · SciPy"
metrics:
  - value: "+21%"
    label: "click lift"
  - value: "~24,800"
    label: "readers in 2010s"
  - value: "0.02"
    label: "p-value"
---

In the early 2010s, Upworthy was one of the most-clicked media sites on the internet — and behind the scenes it A/B tested the headline of almost every story it published, showing different versions to randomly split visitors and counting the clicks. Years later, the complete experiment logs were released as an open research archive. This project takes one experiment from it — two headlines identical except for a **single word** — and asks whether that word genuinely changed reader behaviour, using a chi-square test.

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Model Overview](#02-model-overview)
- [03. Data Overview](#03-data-overview)
- [04. Data Preparation](#04-data-preparation)
- [05. Chi-Square Test](#05-chi-square-test)
  - [Preparing the Table](#cs-table)
  - [Running the Test](#cs-test)
  - [Reading the Verdict](#cs-verdict)
- [06. What the Test Compares](#06-what-the-test-compares)
- [07. Growth & Next Steps](#07-growth--next-steps)

---

## 00. Project Overview

**Context**

Upworthy tested two versions of the same headline: *"For Many Divorced **Women**, Having To Deal With This Every Week Is Simply What Life Is"* and the identical sentence with **Mothers** in place of Women. Visitors were randomly shown one or the other, and the Women version was clicked a little more often. The catch is that two random groups never behave *exactly* alike — one version will always come out slightly ahead by luck alone. The question a headline desk actually needs answered is whether the gap is bigger than luck can explain, and the chi-square test is the standard way to get a yes or no.

**Actions**

I filtered the archive down to that one experiment — two rows, one per headline — built a two-by-two table of clicks and non-clicks for each version, and ran a chi-square test of independence at the 5% significance level.

**Applications**

This is the everyday statistics of publishing. Newsrooms and media companies test headlines, push notifications, and newsletter subject lines on live traffic constantly — and the difference between a real winner and random noise is exactly this test. The same calculation sits inside the testing tools those teams use to decide which version the rest of the audience gets.

**Growth & Next Steps**

The archive holds 61 clean head-to-head headline tests, and the same few lines of code run on all of them would show how often wording genuinely moves readers. Beyond that: a companion example where a one-word change did *nothing* — the archive has one — and reporting the size of the effect with a confidence interval rather than just the yes/no verdict.

---

## 01. Results

| | "Divorced Women" | "Divorced Mothers" |
|---|---|---|
| Readers shown | 11,839 | 12,951 |
| Clicks | 285 | 257 |
| Click rate | 2.41% | 1.98% |

The test returns a chi-square statistic of **5.17** against a critical value of 3.84, with a **p-value of 0.023** — below the 5% threshold, so the null hypothesis is rejected: the two headlines did not perform equally.

**One word moved readers.** The Women version was clicked at 2.41% against 1.98% for Mothers — a gap of about a fifth in relative terms, from changing a single word in an otherwise identical sentence.

**The gap is unlikely to be luck.** A p-value of 0.023 says that if the word truly made no difference, a gap this large would show up in fewer than one in forty runs of the experiment. That is rarer than the 5% bar this project set in advance, so the difference is treated as real.

**The margin is small but it was free.** Across the 24,790 readers in the test, the better word was worth roughly 50 extra clicks — nothing dramatic, but the cost of finding it was zero, and headline tests like this run on every story a site publishes.

---

## 02. Model Overview

The **chi-square test of independence** answers one question about a table of counts: could the pattern have happened by chance? It starts from the null hypothesis that the two headlines perform identically, and works out what the table *should* look like in that world — both versions sharing one overall click rate, so each version's expected clicks are just that shared rate applied to its readers. It then measures how far the observed counts sit from those expected counts, squares the deviations so over- and under-shoots both count, and adds them into a single statistic. A small statistic means the table looks like chance; a large one means it doesn't. The p-value translates the statistic into a probability — how often chance alone would produce a table at least this uneven — and if that probability falls below the significance level chosen before the test (5% here), the null hypothesis is rejected. Nothing about the test is specific to headlines: any two groups and any yes/no outcome fit the same two-by-two table.

---

## 03. Data Overview

The data comes from the **Upworthy Research Archive** — the experiment logs of Upworthy's headline testing between 2013 and 2015, released openly by Cornell University. The exploratory file holds 22,666 headline packages across 4,873 experiments; each row is one version shown in one experiment, with how many people saw it and how many clicked. This analysis uses exactly one experiment: two rows.

| Variable Name | Variable Type | Description |
|---|---|---|
| id | Identifier | Row number — used to pin the two packages in this test |
| clickability_test_id | Identifier | Which experiment a package belongs to |
| headline | Treatment | The wording shown to that group of readers |
| impressions | Outcome | Readers randomly shown this version |
| clicks | Outcome | Readers who clicked it |
| non_clicks | Derived | impressions − clicks, built during preparation |

One note on the archive: it records which image accompanied each headline as an ID, but the image files themselves are not included — so wording is the only treatment this analysis can see.

---

## 04. Data Preparation

There is barely any — which is the point of picking a well-run experiment. The archive arrives pre-aggregated: each row already carries the totals for one version, so there is no person-level table to count up. Preparation is two steps: filter the archive to the two row ids of this experiment, and derive the missing half of the outcome — `non_clicks` — by subtracting clicks from impressions. That yields the two-by-two table of counts the test consumes: clicks and non-clicks for each headline.

---

## 05. Chi-Square Test

### Preparing the Table {#cs-table}

```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2

campaign_data = pd.read_csv("upworthy-archive.csv")

campaign_data = campaign_data.loc[campaign_data["id"].isin([74187,74188])]

campaign_data["non_clicks"] = campaign_data["impressions"] - campaign_data["clicks"]
observed_values = campaign_data[["clicks", "non_clicks"]].values
```

`observed_values` is the two-by-two table: one row per headline, holding clicks and non-clicks.

### Running the Test {#cs-test}

```python
acceptance_criteria = 0.05

chi2_statisctic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)
print(f"chi2 statistic: {chi2_statisctic}\np value: {p_value}")
```

`chi2_contingency` returns the statistic, the p-value, the degrees of freedom (1 for a two-by-two table), and the expected counts under the null. The Yates continuity correction is switched off — it exists for small samples, and the smallest expected count here is around 259.

### Reading the Verdict {#cs-verdict}

```python
critical_value = chi2.ppf(1- acceptance_criteria, dof)

if chi2_statisctic >= critical_value:
    print("We reject the null hypothesis")
else:
    print("We accept the null hypothesis")
```

The statistic (5.17) clears the critical value (3.84), and the p-value (0.023) sits under the 5% acceptance criteria — the same verdict read two ways.

---

## 06. What the Test Compares

The clearest way to see the test's logic is to draw the two tables it compares — the clicks that actually happened against the clicks that *would* have happened if the word made no difference and both versions shared one click rate:

![Observed clicks vs clicks expected under the null]({{ "/img/posts/upworthy_observed_expected.png" | relative_url }})

The Women version collected about 26 clicks more than its share, and the Mothers version about 26 fewer — the same gap seen from both sides. The chi-square statistic is a single number summarising exactly that daylight between the teal and grey bars.

---

## 07. Growth & Next Steps

Concrete improvements queued for the next iteration:

- **Run the whole archive of head-to-heads.** The same two-by-two test in a loop over all 61 two-version experiments would turn one anecdote into a base rate: how often does a headline change actually move readers?
- **Show a null result beside this one.** The archive contains another one-word test — "Schools" vs "Teachers" — where the click rates came out virtually identical. Working that example makes the method's point: the test is what separates the word that mattered from the word that didn't.
- **Report the size, not just the verdict.** A confidence interval on the difference in click rates would say how big the effect plausibly is, which is the number a headline desk would actually act on.

---
