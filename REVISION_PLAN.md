# TRENDSET Revision Plan

This document turns the current review weaknesses into a concrete execution checklist. The goal is to make the paper defensible on three fronts:

1. fairer causal attribution,
2. stronger external validation,
3. clearer and narrower claims.

## A. Core Claim Reset

The revised paper should defend the following claim:

> Neighborhood-conditioned descriptor generation improves routing-oriented utility and boundary clarity under matched inference budgets.

The paper should not imply any stronger claim unless new evidence supports it, especially:

- not "general scientific mapping" beyond local hotspot neighborhoods,
- not "human interpretability is solved",
- not "full neighborhood reward model is the deployed selector" unless the final system actually uses it.

## B. Required New Experiments

### B1. Matched control: Target-only + BoN + RM

Purpose:
- isolate the effect of neighborhood conditioning from candidate search and reranking.

Matched ingredients:
- same backbone,
- same profile prompt family,
- same candidate count `N`,
- same descriptor count `k`,
- same RM reranking budget.

Only removed ingredient:
- neighbor context / neighborhood-conditioned generation.

Deliverable:
- add one new row to the public main table:
  - `Target-only + RM (Qwen2.5-14B)`

Stretch:
- repeat for `Mistral-7B` if compute allows.

### B2. Independent routing evaluator

Purpose:
- show that gains are not unique to one frozen routing proxy.

Required:
- keep current evaluator,
- add at least one independent encoder/router in the main paper.

Recommended:
- `bge-large`
- `e5-large`
- `mpnet`

Deliverable:
- one compact robustness table:
  - methods as rows,
  - `LocalAcc` / `Local-F1` under multiple evaluators as columns.

Goal:
- show method ranking is stable, or state clearly when it is not.

### B3. Human evaluation

Purpose:
- break the "automated proxies all the way down" criticism.

Minimum viable study:
- 30 to 50 blinded pairwise comparisons,
- compare:
  - strongest baseline,
  - TRENDSET,
- raters answer:
  - which set is more readable,
  - which set is more coherent,
  - which better distinguishes the hotspot from nearby topics.

Deliverable:
- one main-paper human study table,
- GPT-5.4 judge stays as supplementary support only.

### B4. Rebuilt ablations under matched settings

Purpose:
- replace weak small-slice or mismatched ablations.

Required:
- same backbone,
- same profile prompt,
- same candidate budget,
- same reranking mode.

#### Neighborhood-size ablation

Compare:
- `Target-only`
- `m = 1`
- `m = 2`
- `m = 3`

Primary outcome:
- utility metrics under the deployed selector.

Judge support:
- optional LLM-as-judge table can remain, but not as the only evidence.

#### Objective-term ablation

Compare:
- `Full`
- `No Routing`
- `No Diversity`
- `No Separation`

Important:
- use the actual deployed reranking path,
- do not use oracle reranking for the main ablation table.

### B5. Neighborhood RM recovery

Purpose:
- rescue the method story if possible.

Target experiment:
- `Qwen2.5 + m=1 + profile + retrained neighborhood RM`

Required steps:
- generate aligned sample pool,
- rebuild generated neighborhood preference pairs,
- retrain neighborhood RM,
- rerun TRENDSET with `reward_mode=neighborhood`.

Sweep dimensions:
- candidate source:
  - `generated`
  - `hybrid`
- reward weights:
  - especially `gamma` for separation

Success criterion:
- neighborhood RM should approach center RM performance closely enough that the paper can still credibly emphasize the composite neighborhood reward.

Fallback:
- if neighborhood RM still underperforms, explicitly demote it in the narrative and present it as a teacher signal rather than the deployed selector.

## C. Split Hygiene and Reproducibility

### C1. Explicit split-hygiene paragraph

Add one paragraph in the main text stating:

- which hotspots are used to build pairwise preference data,
- which split is used for RM early stopping,
- which split is used only for final evaluation,
- that test hotspots are not used in RM training.

If any ambiguity exists in code, fix code first, then write the paragraph.

### C2. Main-text operational details

The main text should explicitly report:

- generator backbone,
- `k`,
- `N`,
- prompt/profile ingredients,
- reward weights,
- margin,
- temperature,
- reranking mode,
- evaluator backbones used in main results.

Prompt templates can stay in appendix, but the knobs above should be stated in the main paper.

## D. Table-by-Table Rewrite Plan

### Table 1: Public benchmark main results

Keep as the primary main-results table.

Rows should include:
- Coverage-Greedy
- TopicGPT
- Qwen2.5-14B
- Mistral-7B
- Qwen3-8B
- DeepSeek-R1
- Gemma-3-4B-IT
- Llama-3.1-8B
- Target-only + RM (new)
- TRENDSET (Qwen2.5-14B)
- TRENDSET (Mistral-7B) if stable enough
- TRENDSET (Qwen3-8B) if stable enough

If some TRENDSET rows remain weak or incomplete:
- move them to appendix instead of weakening the main story.

### Table 2: Human evaluation

Replace or elevate the current LLM-judge emphasis.

Main paper should contain:
- human preference results.

Appendix can contain:
- GPT-5.4 judge prompt,
- rubric,
- LLM-judge replication details.

### Table 3: Ablation

Replace current weak or mismatched version with:

1. matched neighborhood-size ablation,
2. matched objective-term ablation.

If space is tight:
- keep one in the main paper,
- move the other to appendix.

### Table 4: Qualitative examples

Keep strong success cases, but add:
- 2 to 3 balanced failure cases,
- one short paragraph explaining overspecialization vs boundary clarity.

## E. Section-by-Section Text Changes

### Introduction

Tighten framing:
- local neighborhood descriptor learning,
- interpretable but utility-preserving descriptors,
- avoid global-mapping rhetoric.

### Method

Separate clearly:
- generator,
- teacher reward,
- deployed reranker.

If neighborhood RM is not used in final main results:
- do not write as if it is.

### Evaluation

Add:
- split hygiene paragraph,
- matched-budget baseline description,
- independent evaluator description,
- human study description.

Reduce:
- defensive prose,
- implementation-journal style wording,
- phrases like "current reported system" or "for the reported version".

### Discussion

Add one explicit paragraph on:
- lexical distinctiveness vs routing utility,
- why Coverage-Greedy is strong on lexical metrics,
- why utility ranking differs.

Also add:
- benchmark scope limitations,
- local-boundary interpretation of results.

## F. Execution Order

### Phase 1: High-leverage fixes

1. Run `Target-only + BoN + RM`.
2. Retrain and rerun `Qwen2.5 neighborhood RM`.
3. Add one independent routing evaluator.

These three decide whether the method claim stays strong.

### Phase 2: Strengthen evidence

4. Run human pairwise study.
5. Rebuild matched ablations.
6. Add failure-case qualitative analysis.

### Phase 3: Paper rewrite

7. Update Table 1.
8. Replace Table 2 with human evaluation.
9. Rebuild Table 3.
10. Rewrite Intro / Method / Eval accordingly.

## G. Decision Rules

### If neighborhood RM succeeds

Then the paper can claim:
- composite neighborhood reward is operational in both training and deployment.

### If neighborhood RM fails but Target-only + RM loses to TRENDSET

Then the paper can still claim:
- neighborhood-conditioned generation is the key gain,
- target-focused reranking is the stable deployment choice.

### If TRENDSET only beats prompt-only baselines but not Target-only + RM

Then the paper risks collapsing into prompt engineering.
In that case:
- do not submit without either a stronger neighborhood RM or stronger human evidence.

## H. Immediate Next Tasks

1. Finish `Qwen2.5 + m=1 + profile + retrained neighborhood RM`.
2. Finish `Target-only + BoN + RM`.
3. Add second evaluator to the public benchmark.
4. Draft the split-hygiene paragraph from the actual code path.
