# Method Table Explanation

This note explains the method names that appear in the main TRENDSET result tables.

## How to read the methods

The tables mix three types of systems:

- **Extractive baselines**: choose phrases directly from the support documents.
- **Independent generation baselines**: generate descriptors for one hotspot at a time.
- **TRENDSET**: generate with neighborhood context and rerank with the TRENDSET objective.

## Method-by-method explanation

### Coverage-Greedy

A simple heuristic extractive baseline. It selects high-coverage, high-frequency surface phrases from the support set. It is strong on lexical metrics because it reuses exact words that also appear in the references.

### BERTopic

A standard topic-model-style labeling baseline. It appears in the fixed benchmark table as a strong off-the-shelf non-LLM topic labeling system.

### Extractive (Ours)

This is **our extractive baseline**, not the full TRENDSET generator.

The label **"(Ours)"** means that the extraction pipeline is implemented by us inside the same benchmark and evaluation framework as the main method. It uses our semantic hard-negative setup and our selection pipeline, but the output is still a set of extracted phrases rather than generated neighborhood-conditioned descriptors.

So:

- it is **ours** because we built the extractive system for this paper,
- but it is **not TRENDSET**, because it does not perform neighborhood-conditioned generation.

We include it because a strong paper should compare against a strong extractive competitor, not only weak off-the-shelf baselines.

### TopicGPT

A prompt-based topic-labeling baseline. An LLM is prompted to name the hotspot, but it does not use the TRENDSET neighborhood objective.

### Qwen-14B (Zero-shot)

A plain independent generation baseline. Qwen generates descriptors directly for one hotspot using prompting only.

### Mistral-7B (Zero-shot)

The same independent zero-shot generation setup, but with Mistral-7B instead of Qwen-14B.

### Qwen-14B + BoN + RM

This is still an **independent generation** baseline, not TRENDSET.

Its pipeline is:

1. Qwen samples multiple candidate descriptor sets.
2. A reward model scores those candidates.
3. Best-of-\(N\) reranking selects the highest-scoring one.

This is the strongest non-neighborhood generative baseline in the paper.

### TRENDSET

This is the full proposed method.

Its main difference from the independent generator baselines is that it conditions on a **target hotspot plus its nearby semantic neighbors**, then reranks candidates using the TRENDSET objective. The final paper uses the tuned one-neighbor, target-focused, reward-reranked configuration as the main TRENDSET row.

## Why Extractive (Ours) is necessary

If we compared only against Qwen-style generators, a reviewer could argue that we avoided the strongest lexical baselines. `Extractive (Ours)` is included to show that:

- a strong extractive method can win lexical distinctiveness,
- but still lose on the usability and routing side,
- which is exactly part of TRENDSET's main scientific argument.

## Clean grouping

You can mentally group the rows like this:

- **Extractive / topic baselines**: `Coverage-Greedy`, `BERTopic`, `Extractive (Ours)`
- **Independent generative baselines**: `TopicGPT`, `Qwen-14B (Zero-shot)`, `Mistral-7B (Zero-shot)`, `Qwen-14B + BoN + RM`
- **Proposed method**: `TRENDSET`
