#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from sd_robustness.paths import default_cache_root

SHARED_CACHE_ROOT = Path(os.environ.get("SHARED_CACHE_ROOT", str(default_cache_root())))
HF_HOME = Path(os.environ.setdefault("HF_HOME", str(SHARED_CACHE_ROOT / "huggingface")))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("HF_ASSETS_CACHE", str(HF_HOME / "assets"))
os.environ.setdefault("HF_DATASETS_CACHE", str(SHARED_CACHE_ROOT / "hf_datasets"))
os.environ.setdefault("TORCH_HOME", str(SHARED_CACHE_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(SHARED_CACHE_ROOT / "xdg_cache"))
for cache_dir in (
    SHARED_CACHE_ROOT,
    HF_HOME,
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]),
    Path(os.environ["TRANSFORMERS_CACHE"]),
    Path(os.environ["HF_ASSETS_CACHE"]),
    Path(os.environ["HF_DATASETS_CACHE"]),
    Path(os.environ["TORCH_HOME"]),
    Path(os.environ["XDG_CACHE_HOME"]),
):
    cache_dir.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sd_robustness.seed_search import (
    clean_token_text,
    select_calibration_indices,
    format_progress_line,
    run_gradient_guided_seed_search as run_shared_seed_search,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INIT_TEXT = "Complete the next symbol only:\n\\boxed{"
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_TEMPLATE_PATH = (
    REPO_ROOT
    / "external"
    / "hsd"
    / "chain-of-thought-hub"
    / "gsm8k"
    / "lib_prompt"
    / "prompt_original.txt"
)


@dataclass
class CalibrationSample:
    sample_index: int
    question: str
    answer: str
    context_text: str
    context_ids: list[int]
    rollout_ids: list[int]
    rollout_text: str
    draft_context_embeds: torch.Tensor
    draft_context_mask: torch.Tensor
    target_context_embeds: torch.Tensor
    target_context_mask: torch.Tensor
    draft_rollout_input_embeds: torch.Tensor
    draft_rollout_input_mask: torch.Tensor
    target_rollout_input_embeds: torch.Tensor
    target_rollout_input_mask: torch.Tensor
    target_benign_log_probs: torch.Tensor


@dataclass
class AggregatedProjectedMetrics:
    suffix_text: str
    suffix_token_ids: list[int]
    weighted_trajectory_tv: float
    weighted_reverse_kl: float
    weighted_target_preserve_kl: float
    mean_first_alpha_1: float
    mean_collapse_score: float
    mean_rollout_soft_collapse: float


@dataclass
class SeedSearchCandidate:
    suffix_ids: list[int]
    metrics: AggregatedProjectedMetrics


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def rank0_print(ctx: DistributedContext, *parts: object, **kwargs: object) -> None:
    if not ctx.enabled or ctx.rank == 0:
        print(*parts, **kwargs)


def normalized_linear_decay_weights(num_steps: int, *, device: torch.device | str) -> torch.Tensor:
    weights = torch.arange(num_steps, 0, -1, device=device, dtype=torch.float32)
    return weights / weights.sum()


def reduce_rollout_soft_collapse(per_step_values: torch.Tensor, *, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return per_step_values.sum()
    if reduction == "linear":
        weights = normalized_linear_decay_weights(per_step_values.shape[0], device=per_step_values.device)
        return torch.sum(weights * per_step_values)
    return per_step_values.mean()


def reduce_rollout_tv(per_step_values: torch.Tensor, *, reduction: str) -> torch.Tensor:
    if reduction == "linear":
        weights = normalized_linear_decay_weights(per_step_values.shape[0], device=per_step_values.device)
        return torch.sum(weights * per_step_values)
    return per_step_values.mean()


def metrics_to_payload(metrics: AggregatedProjectedMetrics) -> dict[str, object]:
    payload = asdict(metrics)
    payload["suffix_text"] = payload["suffix_text"]
    payload["suffix_token_ids"] = payload["suffix_token_ids"]
    return payload


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def target_first_device(model: AutoModelForCausalLM) -> str:
    if hasattr(model, "hf_device_map"):
        for _, device in model.hf_device_map.items():
            if isinstance(device, int):
                return f"cuda:{device}"
            if isinstance(device, str) and device.startswith("cuda"):
                return device
    return str(next(model.parameters()).device)


def load_models(
    args: argparse.Namespace,
    ctx: DistributedContext,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM, str]:
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.get_vocab() != target_tokenizer.get_vocab():
        raise SystemExit("Draft and target models must share the same tokenizer and vocabulary.")

    if ctx.enabled:
        model_dtype = resolve_torch_dtype(args.tp_dtype)
        local_device = f"cuda:{ctx.local_rank}"
        rank0_print(ctx, f"Loading draft model with tp_plan=auto on {ctx.world_size} ranks ...", flush=True)
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            tp_plan="auto",
            torch_dtype=model_dtype,
        ).eval()
        rank0_print(ctx, "Loading target model with tp_plan=auto and gradient checkpointing ...", flush=True)
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            tp_plan="auto",
            torch_dtype=model_dtype,
        ).train()
        target_model.config.use_cache = False
        try:
            target_model.gradient_checkpointing_enable()
        except Exception:
            rank0_print(ctx, "Warning: gradient checkpointing could not be enabled for tensor-parallel target model.", flush=True)
        for param in draft_model.parameters():
            param.requires_grad_(False)
        for param in target_model.parameters():
            param.requires_grad_(False)
        return tokenizer, draft_model, target_model, local_device

    rank0_print(ctx, f"Loading draft model on {args.draft_device} ...", flush=True)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        device_map={"": args.draft_device},
    ).eval()
    for param in draft_model.parameters():
        param.requires_grad_(False)

    rank0_print(ctx, "Loading target model with device_map=auto and gradient checkpointing ...", flush=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        device_map="auto",
    ).train()
    target_model.config.use_cache = False
    target_model.gradient_checkpointing_enable()
    for param in target_model.parameters():
        param.requires_grad_(False)

    return tokenizer, draft_model, target_model, target_first_device(target_model)


def build_base_ids(tokenizer: AutoTokenizer, text: str) -> list[int]:
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def precompute_base_embeddings(
    model: AutoModelForCausalLM,
    token_ids: list[int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_ids:
        hidden_size = model.get_input_embeddings().weight.shape[1]
        return (
            torch.empty((1, 0, hidden_size), device=device, dtype=model.get_input_embeddings().weight.dtype),
            torch.empty((1, 0), device=device, dtype=torch.long),
        )
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(ids).detach()
    return embeddings, attention_mask


def build_gsm8k_context(
    *,
    question: str,
    prompt_template: str,
    system_prompt: str,
    tokenizer: AutoTokenizer,
) -> str:
    prompt_q = prompt_template + "\nQuestion: " + question + "\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_q},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_task_context(
    *,
    user_prompt: str,
    system_prompt: str,
    tokenizer: AutoTokenizer,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_calibration_rows(
    *,
    dataset_name: str,
    split: str,
) -> tuple[list[dict[str, str]], str]:
    if dataset_name == "gsm8k":
        resolved_split = "train" if split == "auto" else split
        dataset = load_dataset("gsm8k", "main", split=resolved_split)
        rows = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "user_prompt": "",
            }
            for row in dataset
        ]
    elif dataset_name == "humaneval":
        resolved_split = "test" if split == "auto" else split
        dataset = load_dataset(
            "openai_humaneval",
            "openai_humaneval",
            split=resolved_split,
        )
        rows = [
            {
                "question": row["task_id"],
                "answer": row["canonical_solution"],
                "user_prompt": (
                    "Complete the following Python function. Return only executable "
                    f"Python code.\n\n{row['prompt']}"
                ),
            }
            for row in dataset
        ]
    else:
        resolved_split = "train" if split == "auto" else split
        dataset = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            split=resolved_split,
        )
        rows = [
            {
                "question": f"cnn_dailymail_{index}",
                "answer": row["highlights"],
                "user_prompt": (
                    "Summarize the following news article in a concise paragraph.\n\n"
                    f"Article:\n{row['article']}\n\nSummary:"
                ),
            }
            for index, row in enumerate(dataset)
        ]
    return rows, resolved_split


@torch.no_grad()
def generate_target_rollout_ids(
    *,
    context_ids: list[int],
    target_model: AutoModelForCausalLM,
    target_device: str,
    max_new_tokens: int,
    do_sample: bool,
) -> list[int]:
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=target_device)
    attention_mask = torch.ones_like(input_ids)
    outputs = target_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    return outputs[0, input_ids.shape[1] :].detach().cpu().tolist()


def sequence_log_probs_from_embeds(
    *,
    model: AutoModelForCausalLM,
    context_embeds: torch.Tensor,
    context_mask: torch.Tensor,
    suffix_embeds: torch.Tensor,
    rollout_input_embeds: torch.Tensor,
    rollout_input_mask: torch.Tensor,
    target_length: int,
    vocab_size: int,
) -> torch.Tensor:
    suffix_batch = suffix_embeds.unsqueeze(0)
    inputs_embeds = torch.cat([context_embeds, suffix_batch, rollout_input_embeds], dim=1)
    suffix_mask = torch.ones(
        (1, suffix_embeds.shape[0]),
        dtype=context_mask.dtype,
        device=context_mask.device,
    )
    attention_mask = torch.cat([context_mask, suffix_mask, rollout_input_mask], dim=1)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    start = context_embeds.shape[1] + suffix_embeds.shape[0] - 1
    stop = start + target_length
    logits = outputs.logits[0, start:stop, :vocab_size]
    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def exact_first_alpha_for_prompt(
    *,
    prompt_ids: list[int],
    tokenizer: AutoTokenizer,
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    draft_device: str,
    target_device: str,
    vocab_size: int,
) -> tuple[float, float]:
    draft_ids = torch.tensor([prompt_ids], dtype=torch.long, device=draft_device)
    draft_mask = torch.ones_like(draft_ids)
    q_logits = draft_model(input_ids=draft_ids, attention_mask=draft_mask).logits[:, -1, :vocab_size]
    q_log = F.log_softmax(q_logits[0], dim=-1)

    target_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
    target_mask = torch.ones_like(target_ids)
    p_logits = target_model(input_ids=target_ids, attention_mask=target_mask).logits[:, -1, :vocab_size]
    p_log = F.log_softmax(p_logits[0], dim=-1)

    q = q_log.exp()
    p = p_log.exp()
    drafted_token_id = int(torch.argmax(q).item())
    q_selected = float(q[drafted_token_id].item())
    p_selected = float(p[drafted_token_id].item())
    alpha_1 = min(1.0, p_selected / max(q_selected, 1e-12))
    collapse_score = math.log(q_selected + 1e-12) - math.log(p_selected + 1e-12)
    return alpha_1, collapse_score


def truncate_or_pad_suffix(token_ids: list[int], suffix_length: int) -> list[int]:
    clipped = list(token_ids[:suffix_length])
    if len(clipped) < suffix_length:
        clipped.extend([0] * (suffix_length - len(clipped)))
    return clipped


def hard_objective_from_suffix_embeds(
    *,
    draft_suffix_embeds: torch.Tensor,
    target_suffix_embeds: torch.Tensor,
    samples: list[CalibrationSample],
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    vocab_size: int,
    collapse_weight: float,
    collapse_all_reduction: str,
    tv_weight: float,
    revkl_weight: float,
    target_preserve_weight: float,
) -> torch.Tensor:
    mean_tv_terms: list[torch.Tensor] = []
    weighted_revkl_terms: list[torch.Tensor] = []
    early_collapse_terms: list[torch.Tensor] = []
    weighted_target_preserve_terms: list[torch.Tensor] = []
    for sample in samples:
        draft_suffix_embeds_device = draft_suffix_embeds.to(
            device=sample.draft_context_embeds.device,
            dtype=sample.draft_context_embeds.dtype,
        )
        target_suffix_embeds_device = target_suffix_embeds.to(
            device=sample.target_context_embeds.device,
            dtype=sample.target_context_embeds.dtype,
        )
        q_log_seq = sequence_log_probs_from_embeds(
            model=draft_model,
            context_embeds=sample.draft_context_embeds,
            context_mask=sample.draft_context_mask,
            suffix_embeds=draft_suffix_embeds_device,
            rollout_input_embeds=sample.draft_rollout_input_embeds,
            rollout_input_mask=sample.draft_rollout_input_mask,
            target_length=len(sample.rollout_ids),
            vocab_size=vocab_size,
        )
        p_log_seq = sequence_log_probs_from_embeds(
            model=target_model,
            context_embeds=sample.target_context_embeds,
            context_mask=sample.target_context_mask,
            suffix_embeds=target_suffix_embeds_device,
            rollout_input_embeds=sample.target_rollout_input_embeds,
            rollout_input_mask=sample.target_rollout_input_mask,
            target_length=len(sample.rollout_ids),
            vocab_size=vocab_size,
        )
        q_seq = q_log_seq.exp()
        p_seq = p_log_seq.exp()
        benign_log_seq = sample.target_benign_log_probs.to(device=p_log_seq.device, dtype=p_log_seq.dtype)
        benign_seq = benign_log_seq.exp()
        per_step_tv = 0.5 * torch.sum(torch.abs(q_seq - p_seq), dim=-1)
        per_step_revkl = torch.sum(q_seq * (q_log_seq - p_log_seq), dim=-1)
        per_step_soft_collapse = torch.sum(q_seq * F.relu(q_log_seq - p_log_seq), dim=-1)
        per_step_target_preserve_kl = torch.sum(benign_seq * (benign_log_seq - p_log_seq), dim=-1)
        weights = normalized_linear_decay_weights(per_step_tv.shape[0], device=per_step_tv.device)
        mean_tv_terms.append(reduce_rollout_tv(per_step_tv, reduction=collapse_all_reduction))
        weighted_revkl_terms.append(torch.sum(weights * per_step_revkl))
        weighted_target_preserve_terms.append(torch.sum(weights * per_step_target_preserve_kl))
        early_collapse_terms.append(
            reduce_rollout_soft_collapse(
                per_step_soft_collapse,
                reduction=collapse_all_reduction,
            )
        )

    return (
        collapse_weight * torch.stack(early_collapse_terms).mean()
        + tv_weight * torch.stack(mean_tv_terms).mean()
        + revkl_weight * torch.stack(weighted_revkl_terms).mean()
        - target_preserve_weight * torch.stack(weighted_target_preserve_terms).mean()
    )


def gradient_ranked_token_ids(
    *,
    suffix_ids: list[int],
    position: int,
    samples: list[CalibrationSample],
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    draft_embed_weight: torch.Tensor,
    target_embed_weight: torch.Tensor,
    vocab_size: int,
    collapse_weight: float,
    collapse_all_reduction: str,
    tv_weight: float,
    revkl_weight: float,
    target_preserve_weight: float,
    topk: int,
) -> list[int]:
    draft_suffix_embeds = draft_embed_weight[suffix_ids].detach().clone().requires_grad_(True)
    target_suffix_embeds = target_embed_weight[suffix_ids].detach().clone().requires_grad_(True)
    objective = hard_objective_from_suffix_embeds(
        draft_suffix_embeds=draft_suffix_embeds,
        target_suffix_embeds=target_suffix_embeds,
        samples=samples,
        draft_model=draft_model,
        target_model=target_model,
        vocab_size=vocab_size,
        collapse_weight=collapse_weight,
        collapse_all_reduction=collapse_all_reduction,
        tv_weight=tv_weight,
        revkl_weight=revkl_weight,
        target_preserve_weight=target_preserve_weight,
    )
    objective.backward()
    draft_grad = draft_suffix_embeds.grad[position]
    target_grad = target_suffix_embeds.grad[position]

    draft_scores = torch.matmul(
        draft_embed_weight[:vocab_size], draft_grad.to(draft_embed_weight.dtype)
    )
    target_scores = torch.matmul(
        target_embed_weight[:vocab_size], target_grad.to(target_embed_weight.dtype)
    )
    combined_scores = draft_scores.float().cpu() + target_scores.float().cpu()

    current_token = int(suffix_ids[position])
    candidate_count = min(max(topk, 1) + 1, combined_scores.shape[0])
    top_ids = torch.topk(combined_scores, k=candidate_count).indices.tolist()
    ranked_ids = [current_token]
    for token_id in top_ids:
        if token_id not in ranked_ids:
            ranked_ids.append(int(token_id))
        if len(ranked_ids) >= topk + 1:
            break
    return ranked_ids


def run_gradient_guided_seed_search(
    *,
    initial_suffix_ids: list[int],
    samples: list[CalibrationSample],
    tokenizer: AutoTokenizer,
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    draft_device: str,
    target_device: str,
    draft_embed_weight: torch.Tensor,
    target_embed_weight: torch.Tensor,
    vocab_size: int,
    collapse_weight: float,
    collapse_all_reduction: str,
    tv_weight: float,
    revkl_weight: float,
    target_preserve_weight: float,
    beam_size: int,
    topk: int,
    sweeps: int,
    ctx: DistributedContext,
    audit_callback=None,
    progress_callback=None,
    candidate_allowed=None,
) -> list[SeedSearchCandidate]:
    base_metrics = exact_uat_metrics_for_suffix(
        suffix_ids=initial_suffix_ids,
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
    if audit_callback is not None:
        audit_callback("exact")
    initial_candidate = SeedSearchCandidate(
        suffix_ids=list(initial_suffix_ids),
        metrics=base_metrics,
    )

    def objective_from_metrics(metrics: AggregatedProjectedMetrics) -> float:
        return projected_objective_from_metrics(
            metrics=metrics,
            collapse_weight=collapse_weight,
            tv_weight=tv_weight,
            revkl_weight=revkl_weight,
            target_preserve_weight=target_preserve_weight,
        )

    def rank_token_ids(candidate: SeedSearchCandidate, position: int) -> list[int]:
        if audit_callback is not None:
            audit_callback("gradient")
        return gradient_ranked_token_ids(
            suffix_ids=candidate.suffix_ids,
            position=position,
            samples=samples,
            draft_model=draft_model,
            target_model=target_model,
            draft_embed_weight=draft_embed_weight,
            target_embed_weight=target_embed_weight,
            vocab_size=vocab_size,
            collapse_weight=collapse_weight,
            collapse_all_reduction=collapse_all_reduction,
            tv_weight=tv_weight,
            revkl_weight=revkl_weight,
            target_preserve_weight=target_preserve_weight,
            topk=topk,
        )

    def score_suffix(proposal: list[int], _position: int) -> AggregatedProjectedMetrics:
        metrics = exact_uat_metrics_for_suffix(
            suffix_ids=proposal,
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
        if audit_callback is not None:
            audit_callback("exact")
        return metrics

    def build_candidate(proposal: list[int], metrics: AggregatedProjectedMetrics) -> SeedSearchCandidate:
        return SeedSearchCandidate(suffix_ids=proposal, metrics=metrics)

    def log_progress(
        sweep_idx: int,
        sweep_total: int,
        pos_idx: int,
        pos_total: int,
        best: SeedSearchCandidate,
        best_projected_objective: float,
    ) -> None:
        rank0_print(
            ctx,
            format_progress_line(
                "seed",
                [
                    ("sweep", f"{sweep_idx}/{sweep_total}"),
                    ("pos", f"{pos_idx}/{pos_total}"),
                    ("proj_obj", best_projected_objective),
                    ("soft", best.metrics.mean_rollout_soft_collapse),
                    ("tv", best.metrics.weighted_trajectory_tv),
                    ("revkl", best.metrics.weighted_reverse_kl),
                    ("preserve", best.metrics.weighted_target_preserve_kl),
                    ("suffix", clean_token_text(best.metrics.suffix_text)),
                ],
            ),
            flush=True,
        )
        if progress_callback is not None:
            progress_callback(
                stage="gcg",
                step=(sweep_idx - 1) * pos_total + pos_idx,
                total=sweep_total * pos_total,
                best_objective=best_projected_objective,
                best_projected=metrics_to_payload(best.metrics),
            )

    beam = run_shared_seed_search(
        initial_candidate=initial_candidate,
        suffix_length=len(initial_suffix_ids),
        sweeps=sweeps,
        beam_size=beam_size,
        rank_token_ids=rank_token_ids,
        score_suffix=score_suffix,
        build_candidate=build_candidate,
        objective_from_metrics=objective_from_metrics,
        log_progress=log_progress,
        candidate_allowed=candidate_allowed,
    )
    return list(beam)


@torch.no_grad()
def exact_uat_metrics_for_suffix(
    *,
    suffix_ids: list[int],
    samples: list[CalibrationSample],
    tokenizer: AutoTokenizer,
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    draft_device: str,
    target_device: str,
    draft_embed_weight: torch.Tensor,
    target_embed_weight: torch.Tensor,
    vocab_size: int,
    collapse_all_reduction: str = "mean",
) -> AggregatedProjectedMetrics:
    draft_suffix_embeds = draft_embed_weight[suffix_ids]
    target_suffix_embeds = target_embed_weight[suffix_ids]

    mean_tv_values: list[float] = []
    weighted_revkl_values: list[float] = []
    weighted_target_preserve_values: list[float] = []
    alpha_values: list[float] = []
    collapse_values: list[float] = []
    rollout_collapse_values: list[float] = []

    for sample in samples:
        alpha_1, collapse_score = exact_first_alpha_for_prompt(
            prompt_ids=sample.context_ids + suffix_ids,
            tokenizer=tokenizer,
            draft_model=draft_model,
            target_model=target_model,
            draft_device=draft_device,
            target_device=target_device,
            vocab_size=vocab_size,
        )
        alpha_values.append(alpha_1)
        collapse_values.append(collapse_score)

        q_log_seq = sequence_log_probs_from_embeds(
            model=draft_model,
            context_embeds=sample.draft_context_embeds,
            context_mask=sample.draft_context_mask,
            suffix_embeds=draft_suffix_embeds.to(
                device=sample.draft_context_embeds.device,
                dtype=sample.draft_context_embeds.dtype,
            ),
            rollout_input_embeds=sample.draft_rollout_input_embeds,
            rollout_input_mask=sample.draft_rollout_input_mask,
            target_length=len(sample.rollout_ids),
            vocab_size=vocab_size,
        )
        p_log_seq = sequence_log_probs_from_embeds(
            model=target_model,
            context_embeds=sample.target_context_embeds,
            context_mask=sample.target_context_mask,
            suffix_embeds=target_suffix_embeds.to(
                device=sample.target_context_embeds.device,
                dtype=sample.target_context_embeds.dtype,
            ),
            rollout_input_embeds=sample.target_rollout_input_embeds,
            rollout_input_mask=sample.target_rollout_input_mask,
            target_length=len(sample.rollout_ids),
            vocab_size=vocab_size,
        )
        q_seq = q_log_seq.exp()
        p_seq = p_log_seq.exp()
        benign_log_seq = sample.target_benign_log_probs.to(device=p_log_seq.device, dtype=p_log_seq.dtype)
        benign_seq = benign_log_seq.exp()
        per_step_tv = 0.5 * torch.sum(torch.abs(q_seq - p_seq), dim=-1)
        per_step_revkl = torch.sum(q_seq * (q_log_seq - p_log_seq), dim=-1)
        per_step_soft_collapse = torch.sum(q_seq * F.relu(q_log_seq - p_log_seq), dim=-1)
        per_step_target_preserve_kl = torch.sum(benign_seq * (benign_log_seq - p_log_seq), dim=-1)
        weights = normalized_linear_decay_weights(per_step_tv.shape[0], device=per_step_tv.device)
        mean_tv_values.append(float(reduce_rollout_tv(per_step_tv, reduction=collapse_all_reduction).item()))
        weighted_revkl_values.append(float(torch.sum(weights * per_step_revkl).item()))
        weighted_target_preserve_values.append(float(torch.sum(weights * per_step_target_preserve_kl).item()))
        rollout_collapse_values.append(
            float(
                reduce_rollout_soft_collapse(
                    per_step_soft_collapse,
                    reduction=collapse_all_reduction,
                ).item()
            )
        )

    return AggregatedProjectedMetrics(
        suffix_text=tokenizer.decode(suffix_ids),
        suffix_token_ids=list(suffix_ids),
        weighted_trajectory_tv=float(sum(mean_tv_values) / len(mean_tv_values)),
        weighted_reverse_kl=float(sum(weighted_revkl_values) / len(weighted_revkl_values)),
        weighted_target_preserve_kl=float(sum(weighted_target_preserve_values) / len(weighted_target_preserve_values)),
        mean_first_alpha_1=float(sum(alpha_values) / len(alpha_values)),
        mean_collapse_score=float(sum(collapse_values) / len(collapse_values)),
        mean_rollout_soft_collapse=float(sum(rollout_collapse_values) / len(rollout_collapse_values)),
    )


def projected_objective_from_metrics(
    *,
    metrics: AggregatedProjectedMetrics,
    collapse_weight: float,
    tv_weight: float,
    revkl_weight: float,
    target_preserve_weight: float,
) -> float:
    return (
        collapse_weight * metrics.mean_rollout_soft_collapse
        + tv_weight * metrics.weighted_trajectory_tv
        + revkl_weight * metrics.weighted_reverse_kl
        - target_preserve_weight * metrics.weighted_target_preserve_kl
    )


def prepare_calibration_samples(
    *,
    tokenizer: AutoTokenizer,
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    vocab_size: int,
    draft_device: str,
    target_device: str,
    prompt_template: str,
    system_prompt: str,
    calibration_dataset: str = "gsm8k",
    calibration_split: str = "auto",
    calibration_size: int,
    calibration_offset: int,
    calibration_stride: int,
    trajectory_tokens: int,
    rollout_do_sample: bool,
    calibration_indices: list[int] | None = None,
) -> list[CalibrationSample]:
    calibration_rows, resolved_split = load_calibration_rows(
        dataset_name=calibration_dataset,
        split=calibration_split,
    )
    samples: list[CalibrationSample] = []
    indices = select_calibration_indices(len(calibration_rows), calibration_size,
                                         calibration_offset, calibration_stride, calibration_indices)
    for sample_index in indices:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(
            f"[rank{rank}/local{local_rank}] preparing calibration sample {sample_index}",
            flush=True,
        )
        if sample_index >= len(calibration_rows):
            raise SystemExit(
                f"Calibration index {sample_index} exceeds "
                f"{calibration_dataset}/{resolved_split} size {len(calibration_rows)}"
            )
        row = calibration_rows[sample_index]
        if calibration_dataset == "gsm8k":
            context_text = build_gsm8k_context(
                question=row["question"],
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                tokenizer=tokenizer,
            )
        else:
            context_text = build_task_context(
                user_prompt=row["user_prompt"],
                system_prompt=system_prompt,
                tokenizer=tokenizer,
            )
        context_ids = build_base_ids(tokenizer, context_text)
        try:
            rollout_ids = generate_target_rollout_ids(
                context_ids=context_ids,
                target_model=target_model,
                target_device=target_device,
                max_new_tokens=trajectory_tokens,
                do_sample=rollout_do_sample,
            )
        except Exception:
            print(
                f"[rank{rank}/local{local_rank}] failed during target rollout generation "
                f"for calibration sample {sample_index}",
                file=sys.stderr,
                flush=True,
            )
            raise
        if not rollout_ids:
            raise SystemExit(f"Target rollout was empty for calibration sample {sample_index}")

        rollout_input_ids = rollout_ids[:-1]
        try:
            draft_context_embeds, draft_context_mask = precompute_base_embeddings(draft_model, context_ids, draft_device)
            target_context_embeds, target_context_mask = precompute_base_embeddings(target_model, context_ids, target_device)
            draft_rollout_input_embeds, draft_rollout_input_mask = precompute_base_embeddings(
                draft_model, rollout_input_ids, draft_device
            )
            target_rollout_input_embeds, target_rollout_input_mask = precompute_base_embeddings(
                target_model, rollout_input_ids, target_device
            )
            empty_target_suffix_embeds = torch.empty(
                (0, target_context_embeds.shape[-1]),
                device=target_context_embeds.device,
                dtype=target_context_embeds.dtype,
            )
            target_benign_log_probs = sequence_log_probs_from_embeds(
                model=target_model,
                context_embeds=target_context_embeds,
                context_mask=target_context_mask,
                suffix_embeds=empty_target_suffix_embeds,
                rollout_input_embeds=target_rollout_input_embeds,
                rollout_input_mask=target_rollout_input_mask,
                target_length=len(rollout_ids),
                vocab_size=vocab_size,
            ).detach()
        except Exception:
            print(
                f"[rank{rank}/local{local_rank}] failed while building embeddings "
                f"for calibration sample {sample_index}",
                file=sys.stderr,
                flush=True,
            )
            raise

        samples.append(
            CalibrationSample(
                sample_index=sample_index,
                question=row["question"],
                answer=row["answer"],
                context_text=context_text,
                context_ids=context_ids,
                rollout_ids=rollout_ids,
                rollout_text=tokenizer.decode(rollout_ids),
                draft_context_embeds=draft_context_embeds,
                draft_context_mask=draft_context_mask,
                target_context_embeds=target_context_embeds,
                target_context_mask=target_context_mask,
                draft_rollout_input_embeds=draft_rollout_input_embeds,
                draft_rollout_input_mask=draft_rollout_input_mask,
                target_rollout_input_embeds=target_rollout_input_embeds,
                target_rollout_input_mask=target_rollout_input_mask,
                target_benign_log_probs=target_benign_log_probs,
            )
        )
    return samples
