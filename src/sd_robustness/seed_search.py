from __future__ import annotations

from typing import Any, Callable, Protocol


class SeedSearchCandidateLike(Protocol):
    suffix_ids: list[int]
    metrics: Any


def clean_token_text(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def format_progress_line(tag: str, fields: list[tuple[str, object]]) -> str:
    formatted_fields: list[str] = []
    for key, value in fields:
        if isinstance(value, float):
            formatted_fields.append(f"{key}={value:8.4f}")
        else:
            formatted_fields.append(f"{key}={value}")
    return f"[{tag:<10}] " + " | ".join(formatted_fields)


def run_gradient_guided_seed_search(
    *,
    initial_candidate: SeedSearchCandidateLike,
    suffix_length: int,
    sweeps: int,
    beam_size: int,
    rank_token_ids: Callable[[SeedSearchCandidateLike, int], list[int]],
    score_suffix: Callable[[list[int], int], Any],
    build_candidate: Callable[[list[int], Any], SeedSearchCandidateLike],
    objective_from_metrics: Callable[[Any], float],
    log_progress: Callable[[int, int, int, int, SeedSearchCandidateLike, float], None],
    candidate_allowed: Callable[[list[int]], bool] | None = None,
) -> list[SeedSearchCandidateLike]:
    beam: list[SeedSearchCandidateLike] = [initial_candidate]

    for sweep in range(sweeps):
        for position in range(suffix_length):
            candidates_by_suffix: dict[tuple[int, ...], SeedSearchCandidateLike] = {
                tuple(item.suffix_ids): item for item in beam
            }
            for beam_item in beam:
                ranked_ids = rank_token_ids(beam_item, position)
                for token_id in ranked_ids:
                    proposal = list(beam_item.suffix_ids)
                    proposal[position] = int(token_id)
                    proposal_key = tuple(proposal)
                    if proposal_key in candidates_by_suffix:
                        continue
                    if candidate_allowed is not None and not candidate_allowed(proposal):
                        continue
                    metrics = score_suffix(proposal, position)
                    candidates_by_suffix[proposal_key] = build_candidate(proposal, metrics)

            beam = sorted(
                candidates_by_suffix.values(),
                key=lambda item: objective_from_metrics(item.metrics),
                reverse=True,
            )[:beam_size]
            best = beam[0]
            best_objective = objective_from_metrics(best.metrics)
            log_progress(sweep + 1, sweeps, position + 1, suffix_length, best, best_objective)

    return beam


def select_calibration_indices(total_rows, size, offset, stride, explicit=None):
    if size <= 0 or offset < 0 or stride <= 0:
        raise ValueError("Invalid calibration size/offset/stride")
    indices = list(explicit) if explicit is not None else [offset + i*stride for i in range(size)]
    if len(indices) != size or len(set(indices)) != size:
        raise ValueError("Calibration IDs must be unique and match calibration size")
    if any(type(i) is not int or not 0 <= i < total_rows for i in indices):
        raise ValueError("Calibration ID outside dataset")
    return indices
