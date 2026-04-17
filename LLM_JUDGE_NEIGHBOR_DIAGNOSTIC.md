# LLM Judge Diagnostic for Neighborhood Size

Backbone:
- `TRENDSET (Qwen2.5-14B)`

Runs compared:
- `Qwen baseline`: `/home/runw/Project/cosmospaper/research/trendset/results/llm_free_generate_open_large_qwen14b_profileonly_v1/prompt_outputs.csv`
- `m=1`: `review_subset_neighbors1_20260407_162317`
- `m=2`: `review_subset_neighbors2_20260407_162317`
- `m=3`: `review_subset_neighbors3_20260407_162317`

Matched cases:
- `addiction medicine`
- `astro-ph.EP`
- `astro-ph.SR`
- `biochemistry`
- `cancer biology`
- `cardiovascular medicine`
- `cell biology`
- `cond-mat`

Questionnaire used for each target hotspot:
1. `Readability`: Are the target descriptors understandable as subfield-level phrases without reading the support documents?
2. `Coherence`: Do the five descriptors form a coherent descriptor set rather than a loose list of paper fragments?
3. `Boundary clarity`: Relative to the listed neighboring hotspots, do the target descriptors make the focal hotspot easy to tell apart?
4. `Presentation readiness`: Would this descriptor set be acceptable for direct presentation in a scientific map?

Scoring:
- Each question is scored from `1` to `5`
- An overall preferred `m` is selected for each hotspot

## Per-case preferences

| Hotspot | Preferred `m` | Short rationale |
| --- | --- | --- |
| addiction medicine | `1` | Keeps the legal/withdrawal/genetic axes while staying specific and readable. |
| astro-ph.EP | `3` | Produces the most subfield-like astronomical concepts instead of repeating object-specific title fragments. |
| astro-ph.SR | `1` | Gives the clearest solar-physics boundary against nearby space-physics labels. |
| biochemistry | `1` | Best balance between biochemical specificity and descriptor-set coherence. |
| cancer biology | `1` | Most readable and presentation-ready set; larger neighborhoods collapse into repetitive fragments. |
| cardiovascular medicine | `3` | Broadens the set into stable cardiology concepts without obvious phrase collapse. |
| cell biology | `1` | Strongest full-set coherence; larger neighborhoods fragment into incomplete phrases. |
| cond-mat | `1` | Most compact and readable condensed-matter summary under the matched subset. |

## Aggregate judge scores

Average score across the eight matched hotspots:

| Variant | Readability | Coherence | Boundary clarity | Presentation readiness | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen baseline` | `3.3` | `3.2` | `2.4` | `3.0` | `3.0` |
| `m=1` | `4.3` | `4.1` | `4.5` | `3.9` | `4.2` |
| `m=2` | `3.7` | `3.5` | `3.7` | `3.3` | `3.6` |
| `m=3` | `3.6` | `3.3` | `4.0` | `3.3` | `3.6` |

Summary:
- `m=1` is the best overall neighborhood size under this judge rubric.
- `m=2` and `m=3` are both clearly stronger than the matched Qwen baseline under the same rubric.
- `m=2` is the more coherent of the two larger-neighborhood settings, while `m=3` is stronger on boundary abstraction.
- The matched Qwen baseline often reads cleanly at first glance, but across the eight cases it is too generic to provide strong hotspot boundaries and is less coherent once the descriptor set is read as a whole.
