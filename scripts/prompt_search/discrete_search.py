#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import optimize_universal_suffix as uat


@dataclass
class SuffixCandidate:
    suffix_ids: list[int]
    metrics: uat.AggregatedProjectedMetrics


@dataclass
class SearchAudit:
    exact_objective_evaluations: int = 0
    gradient_evaluations: int = 0


def make_rng(seed: int) -> random.Random:
    rng = random.Random(seed)
    return rng


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def projected_objective(
    metrics: uat.AggregatedProjectedMetrics,
    *,
    collapse_weight: float,
    tv_weight: float,
    revkl_weight: float,
    target_preserve_weight: float = 0.0,
) -> float:
    return uat.projected_objective_from_metrics(
        metrics=metrics,
        collapse_weight=collapse_weight,
        tv_weight=tv_weight,
        revkl_weight=revkl_weight,
        target_preserve_weight=target_preserve_weight,
    )


def better(
    candidate: uat.AggregatedProjectedMetrics,
    incumbent: uat.AggregatedProjectedMetrics | None,
    *,
    collapse_weight: float,
    tv_weight: float,
    revkl_weight: float,
    target_preserve_weight: float = 0.0,
) -> bool:
    if incumbent is None:
        return True
    return projected_objective(
        candidate,
        collapse_weight=collapse_weight,
        tv_weight=tv_weight,
        revkl_weight=revkl_weight,
        target_preserve_weight=target_preserve_weight,
    ) > projected_objective(
        incumbent,
        collapse_weight=collapse_weight,
        tv_weight=tv_weight,
        revkl_weight=revkl_weight,
        target_preserve_weight=target_preserve_weight,
    )


def score_suffix(
    suffix_ids: list[int],
    *,
    samples,
    tokenizer,
    draft_model,
    target_model,
    draft_device,
    target_device,
    draft_embed_weight,
    target_embed_weight,
    vocab_size: int,
    collapse_all_reduction: str,
) -> uat.AggregatedProjectedMetrics:
    return uat.exact_uat_metrics_for_suffix(
        suffix_ids=suffix_ids,
        samples=samples,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        draft_device=draft_device,
        target_device=target_device,
        draft_embed_weight=draft_embed_weight,
        target_embed_weight=target_embed_weight,
        vocab_size=vocab_size,
        collapse_all_reduction=collapse_all_reduction,
    )


def gcg_search(
    *,
    initial_suffix_ids: list[int],
    samples,
    tokenizer,
    draft_model,
    target_model,
    draft_device,
    target_device,
    draft_embed_weight,
    target_embed_weight,
    vocab_size: int,
    collapse_weight: float,
    collapse_all_reduction: str,
    tv_weight: float,
    revkl_weight: float,
    target_preserve_weight: float,
    beam_size: int,
    topk: int,
    sweeps: int,
    audit: SearchAudit | None = None,
    progress_callback=None,
    candidate_allowed=None,
) -> list[SuffixCandidate]:
    def audit_callback(kind: str) -> None:
        if audit is None:
            return
        if kind == "exact":
            audit.exact_objective_evaluations += 1
        elif kind == "gradient":
            audit.gradient_evaluations += 1
        else:
            raise ValueError(f"unknown audit event: {kind}")

    beam = uat.run_gradient_guided_seed_search(
        initial_suffix_ids=initial_suffix_ids,
        samples=samples,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        draft_device=draft_device,
        target_device=target_device,
        draft_embed_weight=draft_embed_weight,
        target_embed_weight=target_embed_weight,
        vocab_size=vocab_size,
        collapse_weight=collapse_weight,
        collapse_all_reduction=collapse_all_reduction,
        tv_weight=tv_weight,
        revkl_weight=revkl_weight,
        target_preserve_weight=target_preserve_weight,
        beam_size=beam_size,
        topk=topk,
        sweeps=sweeps,
        ctx=uat.DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0),
        audit_callback=audit_callback,
        progress_callback=progress_callback,
        candidate_allowed=candidate_allowed,
    )
    return [SuffixCandidate(suffix_ids=item.suffix_ids, metrics=item.metrics) for item in beam]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADSD discrete suffix search")
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--draft-model', required=True)
    parser.add_argument('--target-model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--progress-output', type=Path, required=True)
    cli = parser.parse_args()
    values = json.loads(cli.config.read_text())
    values.update(vars(cli))
    values['prompt_template_file'] = uat.DEFAULT_TEMPLATE_PATH
    values['calibration_indices'] = None
    return argparse.Namespace(**values)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = make_rng(args.seed)
    audit = SearchAudit()
    started_at = time.time()
    progress_state = {
        "status": "initializing",
        "method": args.method,
        "seed": args.seed,
        "started_at_unix": started_at,
    }

    def record_progress(**updates) -> None:
        progress_state.update(updates)
        progress_state["updated_at_unix"] = time.time()
        progress_state["search_audit"] = asdict(audit)
        if args.progress_output:
            atomic_write_json(args.progress_output, progress_state)

    def handle_termination(signum, _frame) -> None:
        record_progress(status="preempted", signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)
    record_progress(status="loading_models")
    ctx = uat.DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0)

    prompt_template = (
        args.prompt_template_file.read_text(encoding="utf-8")
        if args.calibration_dataset == "gsm8k"
        else ""
    )
    tokenizer, draft_model, target_model, target_device = uat.load_models(args, ctx)
    draft_device = str(next(draft_model.parameters()).device)
    vocab_size = min(draft_model.config.vocab_size, target_model.config.vocab_size)
    draft_embed_weight = draft_model.get_input_embeddings().weight[:vocab_size].detach()
    target_embed_weight = target_model.get_input_embeddings().weight[:vocab_size].detach()

    samples = uat.prepare_calibration_samples(
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        draft_device=draft_device,
        target_device=target_device,
        vocab_size=vocab_size,
        prompt_template=prompt_template,
        system_prompt=args.system_prompt,
        calibration_dataset=args.calibration_dataset,
        calibration_split=args.calibration_split,
        calibration_size=args.calibration_size,
        calibration_offset=args.calibration_offset,
        calibration_stride=args.calibration_stride,
        trajectory_tokens=args.trajectory_tokens,
        rollout_do_sample=args.rollout_do_sample,
        calibration_indices=(json.loads(args.calibration_indices.read_text())["indices"]
                             if args.calibration_indices else None),
    )

    init_suffix_ids = uat.truncate_or_pad_suffix(
        tokenizer.encode(args.init_text, add_special_tokens=False),
        args.suffix_length,
    )
    special_ids = set(tokenizer.all_special_ids)
    allowed_tokens = [i for i in range(min(vocab_size, len(tokenizer))) if i not in special_ids]
    def candidate_allowed(ids):
        if any(i in special_ids for i in ids):
            return False
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=False) != ids:
            return False
        return all(tokenizer.encode(sample.context_text + text, add_special_tokens=False)
                   == sample.context_ids + ids for sample in samples)

    if args.init_random:
        for attempt in range(10000):
            init_suffix_ids = [rng.choice(allowed_tokens) for _ in range(args.suffix_length)]
            if candidate_allowed(init_suffix_ids):
                break
        else:
            raise ValueError("Unable to draw a text-roundtripping random initialization")
    if args.require_text_roundtrip and not candidate_allowed(init_suffix_ids):
        raise ValueError("Initial suffix does not match the actual text-evaluation prompt")
    record_progress(initial_suffix_ids=init_suffix_ids,
                    initial_suffix_text=tokenizer.decode(init_suffix_ids),
                    initialization="random_ordinary_tokens_rejection_sampled" if args.init_random else "fixed_text",
                    calibration_manifest=[{"index": x.sample_index, "question": x.question,
                                           "rollout_ids": x.rollout_ids} for x in samples])

    compare_suffix_ids = uat.truncate_or_pad_suffix(
        tokenizer.encode(args.compare_text, add_special_tokens=False),
        args.suffix_length,
    )

    def score_fn(suffix_ids: list[int]) -> uat.AggregatedProjectedMetrics:
        metrics = score_suffix(
            suffix_ids,
            samples=samples,
            tokenizer=tokenizer,
            draft_model=draft_model,
            target_model=target_model,
            draft_device=draft_device,
            target_device=target_device,
            draft_embed_weight=draft_embed_weight,
            target_embed_weight=target_embed_weight,
            vocab_size=vocab_size,
            collapse_all_reduction=args.collapse_all_reduction,
        )
        audit.exact_objective_evaluations += 1
        return metrics

    diagnostics = None
    if args.diagnose_objective:
        # Independent hard-autograd and exact-metric paths on one calibration prompt.
        ds = draft_embed_weight[init_suffix_ids].detach().clone().requires_grad_(True)
        ts = target_embed_weight[init_suffix_ids].detach().clone().requires_grad_(True)
        objective = uat.hard_objective_from_suffix_embeds(
            draft_suffix_embeds=ds, target_suffix_embeds=ts, samples=samples[:1],
            draft_model=draft_model, target_model=target_model, vocab_size=vocab_size,
            collapse_weight=args.collapse_weight, collapse_all_reduction=args.collapse_all_reduction,
            tv_weight=args.tv_weight, revkl_weight=args.revkl_weight,
            target_preserve_weight=args.target_preserve_weight)
        objective.backward()
        exact = uat.exact_uat_metrics_for_suffix(
            suffix_ids=init_suffix_ids, samples=samples[:1], tokenizer=tokenizer,
            draft_model=draft_model, target_model=target_model,
            draft_device=draft_device, target_device=target_device,
            draft_embed_weight=draft_embed_weight, target_embed_weight=target_embed_weight,
            vocab_size=vocab_size, collapse_all_reduction=args.collapse_all_reduction)
        reference = uat.projected_objective_from_metrics(
            metrics=exact, collapse_weight=args.collapse_weight, tv_weight=args.tv_weight,
            revkl_weight=args.revkl_weight, target_preserve_weight=args.target_preserve_weight)
        diagnostics = {"hard_objective": float(objective.detach()), "exact_objective": reference,
                       "absolute_difference": abs(float(objective.detach())-reference),
                       "draft_gradient_norm": float(ds.grad.float().norm()),
                       "target_gradient_norm": float(ts.grad.float().norm()),
                       "gradients_finite": bool(torch.isfinite(ds.grad).all() and torch.isfinite(ts.grad).all())}
        record_progress(objective_diagnostics=diagnostics)
        if not diagnostics["gradients_finite"] or diagnostics["draft_gradient_norm"] <= 0 or diagnostics["target_gradient_norm"] <= 0:
            raise ValueError("Nonfinite or zero suffix gradients")
        if diagnostics["absolute_difference"] > .02 * max(1., abs(reference)):
            raise ValueError("Hard and exact search objectives disagree beyond FP16 tolerance")
        del objective, ds, ts

    reference_metrics = score_fn(compare_suffix_ids)
    record_progress(status="searching")
    ranked = gcg_search(
        initial_suffix_ids=init_suffix_ids,
        samples=samples,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        draft_device=draft_device,
        target_device=target_device,
        draft_embed_weight=draft_embed_weight,
        target_embed_weight=target_embed_weight,
        vocab_size=vocab_size,
        collapse_weight=args.collapse_weight,
        collapse_all_reduction=args.collapse_all_reduction,
        tv_weight=args.tv_weight,
        revkl_weight=args.revkl_weight,
        target_preserve_weight=args.target_preserve_weight,
        beam_size=args.beam_size,
        topk=args.topk,
        sweeps=args.sweeps,
        audit=audit,
        progress_callback=record_progress,
        candidate_allowed=candidate_allowed if args.require_text_roundtrip else None,
    )

    best = ranked[0]
    payload = {
        "method": args.method,
        "args": json_safe(vars(args)),
        "suffix_length": args.suffix_length,
        "search_audit": asdict(audit),
        "initial_suffix_ids": init_suffix_ids,
        "objective_diagnostics": diagnostics,
        "deployment_text_roundtrip_valid": (candidate_allowed(best.suffix_ids)
            and tokenizer.encode(best.metrics.suffix_text, add_special_tokens=False) == best.suffix_ids
            and all(tokenizer.encode(x.context_text + best.metrics.suffix_text, add_special_tokens=False)
                    == x.context_ids + best.suffix_ids for x in samples)),
        "calibration_manifest": progress_state["calibration_manifest"],
        "reference": uat.metrics_to_payload(reference_metrics),
        "best_projected": uat.metrics_to_payload(best.metrics),
        "beats_reference": better(
            best.metrics,
            reference_metrics,
            collapse_weight=args.collapse_weight,
            tv_weight=args.tv_weight,
            revkl_weight=args.revkl_weight,
            target_preserve_weight=args.target_preserve_weight,
        ),
        "top_candidates": [
            {
                "rank": idx + 1,
                "suffix_ids": item.suffix_ids,
                "metrics": uat.metrics_to_payload(item.metrics),
            }
            for idx, item in enumerate(ranked)
        ],
    }

    if args.output:
        atomic_write_json(args.output, payload)

    record_progress(
        status="complete",
        elapsed_seconds=time.time() - started_at,
        output=str(args.output) if args.output else None,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
