# TRENDSET Experiment Redesign

## Problem With The Current Open Benchmark

The current `Inspec/SemEval2010/NUS/Krapivin` study is a document-level keyphrase extraction transfer test.
That experiment is not tightly aligned with the paper's main claim, which is **cluster/hotspot descriptor-set construction**.

It should be:
- removed from the main paper body or moved to the appendix,
- reframed as a weak auxiliary transfer result if kept at all,
- replaced in the main evaluation by **cluster-level, label-derived, scientific benchmarks**.

## Revised Evaluation Story

The main paper should answer four questions:

1. Do standard baselines fail to produce descriptor sets that are **jointly distinctive, complementary, and valid**?
2. Does TRENDSET beat strong baselines under a **controlled same-pool comparison**?
3. Does TRENDSET remain effective when clusters are **hard to separate** or **slightly noisy**?
4. Does TRENDSET transfer beyond CosmosPaper to **open scientific cluster benchmarks**?

## Recommended Open Benchmark Replacement

### A. MTEB scientific clustering datasets

Use:
- `ArXivClusteringP2P/S2S`
- `BiorxivClusteringP2P/S2S`
- `MedrxivClusteringP2P/S2S`

Why this is better:
- the unit is a **cluster/category of papers**, not a single document;
- labels come from **human-assigned subject categories**;
- MTEB includes **fine-grained within-main-category splits**, which gives us true hard negatives;
- the data is scientific and public, so it is much closer to the TRENDSET claim.

Suggested benchmark construction:
- group documents by gold category;
- sample a support set per category to construct descriptors;
- evaluate on held-out documents from the same categories;
- use same-main-category negatives for the fine-grained setting.

### B. SciHTC as a hierarchical scientific label benchmark

Use SciHTC (Sadat and Caragea, EMNLP 2022), which contains `186,160` papers and `1,233` ACM CCS categories.

Why this is useful:
- categories are scientific and hierarchical;
- we can evaluate both coarse and fine descriptor quality;
- it directly addresses the criticism that `48` hotspots is too small.

Suggested benchmark construction:
- form pseudo-hotspots from leaf categories and internal categories;
- construct descriptors from sampled papers under each category;
- evaluate document assignment and category identification on held-out papers.

## Revised Main Experiments

### 1. Controlled descriptor-set benchmark on fixed hotspots

This remains the main experiment.
But the baselines should be upgraded and controlled more tightly.

Use one fixed candidate pool per hotspot and compare:
- `Top-K prior`
- `MMR`
- `MaxSum`
- `Submodular coverage/facility-location`
- `LLM-select-from-pool`
- `LLM-generate-free-form`
- `Best-of-N reward rerank`
- `TRENDSET`

The key rule is:
**at least several baselines must use the same candidate pool and the same information budget.**

This directly addresses the apples-to-oranges critique.

### 2. Open scientific label-derived benchmark

Run TRENDSET and baselines on:
- MTEB scientific clustering datasets
- and/or SciHTC-derived pseudo-hotspots

This should replace the current document-keyphrase transfer table in the main paper.

### 3. Cluster-noise robustness

Inject `10% / 20% / 30%` semantically adjacent off-topic documents into each hotspot.

Report how descriptor quality degrades for:
- BERTopic labels
- LLM labels
- same-pool baselines
- TRENDSET

This directly addresses the critique that the method relies entirely on good upstream clustering.

### 4. Hard-negative sensitivity

Compare three negative constructions:
- random corpus negatives
- same venue/year negatives
- semantic nearest-neighbor negatives

If the paper keeps a discriminative mining module, the main reported setup should use:
- **semantic nearest-neighbor negatives**

Possible construction:
- embed cluster centroids with SPECTER2 or SciBERT;
- for hotspot `k`, choose negatives from the top-`m` nearest other hotspots;
- sample held-out papers from those nearest hotspots.

### 5. Reward sensitivity

The paper needs a real sensitivity study.

Minimum:
- sweep each reward coefficient up/down around the default;
- show rank stability and score stability;
- include a small Pareto-front plot.

Best practice:
- tune weights on a dev split only;
- freeze them before the final test split.

## Revised Baseline Suite

### Core non-LLM baselines

- YAKE
- TopicRank
- KeyBERT
- KeyBERT-MMR
- BERTopic
- TopicGPT
- same-pool submodular selector

### Critical LLM baselines

At least one frontier LLM baseline must be added:
- `GPT-4o` or a newer OpenAI frontier model if available
- `Claude 3.5/3.7` if accessible

And one open LLM baseline:
- `Qwen2.5-72B-Instruct`
- or another strong openly available instruction model

Recommended prompt variants:
- **free-form generation**: generate exactly `k` distinctive descriptor phrases
- **pool-constrained selection**: choose `k` phrases from the same candidate pool

The pool-constrained LLM baseline is especially important because it removes the unfair advantage of unrestricted generation.

## Revised Metrics

### Distinctiveness

Measure whether a descriptor set separates the target hotspot from semantically adjacent hotspots.

Use:
- `F1`
- `Balanced Accuracy`
- `AUPRC`

on **held-out target vs semantic hard-negative documents**.

### Complementarity

Measure whether each phrase contributes a new useful aspect.

Use:
- `UniqueCov`: unique positive-document coverage
- `MargGain`: marginal positive coverage gain
- `LexRed`: lexical redundancy
- `SemRed`: semantic redundancy

### Validity

Use:
- artifact rate
- malformed phrase rate
- human or LLM-judge well-formedness

### Use-oriented label quality

Borrow from TENOR and PROXANN:
- infer a category from descriptors plus a few support documents;
- judge whether held-out documents fit the inferred category;
- rank which held-out documents are most representative.

This is much closer to how humans actually use topic labels.

## Recommended Main-Body Tables

### Table 1: Controlled same-pool comparison on fixed hotspots

Columns:
- F1
- Balanced Acc
- UniqueCov
- MargGain
- LexRed
- SemRed
- ArtifactRate

Methods:
- same-pool baselines
- LLM pool-select
- TRENDSET

### Table 2: Open scientific label-derived benchmark

Datasets:
- ArXiv fine-grained
- ArXiv coarse
- BioRxiv
- MedRxiv
- or SciHTC coarse/fine

Columns:
- category assignment accuracy / macro-F1
- category ranking nDCG
- held-out doc assignment accuracy

### Table 3: Robustness to cluster noise

Rows:
- noise levels `0/10/20/30%`

Methods:
- BERTopic
- LLM baseline
- TRENDSET

### Table 4: Reward ablation and sensitivity

Rows:
- remove complementarity
- remove validity
- remove discriminative term
- coefficient sweeps

## Recommended Figures

### Figure 1

Current motivation figure:
- keep the baseline trade-off idea,
- but replot with **semantic hard negatives**,
- and include one LLM baseline.

### Figure 2

A robustness plot:
- x-axis = cluster noise level
- y-axis = distinctiveness or use-oriented score

### Figure 3

A Pareto plot:
- x-axis = redundancy
- y-axis = category assignment accuracy or F1

This is better than a document-keyphrase transfer scatter plot.

## Recommendation For The Paper Narrative

The main paper should say:

- TRENDSET is about **descriptor-set optimization for clusters**
- its strongest evidence should come from **cluster-level scientific benchmarks**
- the open benchmark should test **label utility**, not single-document keyphrase extraction

The old keyphrase transfer section can be:
- moved to the appendix,
- shortened to one paragraph,
- or removed entirely.

## Sources To Cite In The Revision

- MTEB clustering datasets and labels from human categories: Muennighoff et al., EACL 2023
- SciHTC: Sadat and Caragea, EMNLP 2022
- TENOR: Li et al., EACL 2024
- PROXANN: ACL 2025
- TopicGPT: NAACL 2024
- Topic labeling / label evaluation references already in the draft
