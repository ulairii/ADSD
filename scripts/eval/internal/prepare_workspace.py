#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_REPO = PROJECT_ROOT / "external" / "hsd"
DEFAULT_WORKSPACE = PROJECT_ROOT / "experiments" / "hsd_inference" / "workspace"


def copy_workspace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", ".idea", "__pycache__", "*.pyc"),
    )


def replace_once_or_skip(text: str, old: str, new: str, path: Path) -> str:
    if new in text:
        return text
    if old in text:
        return text.replace(old, new, 1)
    raise RuntimeError(f"Expected text not found while patching {path}: {old}")


def patch_eval_script(path: Path) -> None:
    text = path.read_text()

    replacements = {
        "os.environ['HF_HOME'] = '/home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/hf_home'": (
            "from pathlib import Path\n"
            "PROMPT_DIR = Path(__file__).resolve().parent / 'lib_prompt'\n"
            'os.environ.setdefault("HF_HOME", os.environ.get("SHARED_CACHE_ROOT", "/project/mpese/tigersec/runw/workdirs/speculative-decoding-robustness/cache") + "/huggingface")'
        ),
        "parser.add_argument('--target-model',  default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='must be complex or original')": (
            "parser.add_argument('--target-model',  default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='must be complex or original')\n"
            "parser.add_argument('--draft-model', default='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8')\n"
            "parser.add_argument('--num-samples', type=int, default=None)\n"
            "parser.add_argument('--prompt-suffix', dest='prompt_suffix', default='')\n"
            "parser.add_argument('--prompt-tag', default='')\n"
            "parser.add_argument('--seed', type=int, default=2027)\n"
            "parser.add_argument('--max-new-tokens', type=int, default=512)\n"
            "parser.add_argument('--min-new-tokens', type=int, default=0)\n"
            "parser.add_argument('--sample-offset', type=int, default=0)"
        ),
        "args = parser.parse_args()": (
            "args = parser.parse_args()\n"
            "random.seed(args.seed)\n"
            "np.random.seed(args.seed)\n"
            "torch.manual_seed(args.seed)\n"
            "if torch.cuda.is_available():\n"
            "    torch.cuda.manual_seed_all(args.seed)"
        ),
        'model1_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"': "model1_name = args.draft_model",
        'model_size = model2_name.split("/")[1].split("-")[1]': (
            'model_ref = os.path.basename(model2_name.rstrip("/"))\n'
            'size_match = re.search(r"(\\d+(?:\\.\\d+)?B)", model_ref)\n'
            'if size_match is None:\n'
            '    raise ValueError(f"Could not infer model size from target model path: {model2_name}")\n'
            'model_size = size_match.group(1)'
        ),
        "num_samples = len(gsm8k_test['question'])//5": (
            "num_samples = args.num_samples if args.num_samples is not None else len(gsm8k_test['question'])//5\n"
            "if args.sample_offset < 0 or args.sample_offset + num_samples > len(gsm8k_test['question']):\n"
            "    raise ValueError('sample-offset/num-samples exceed the GSM8K split')"
        ),
        "for q, a in tqdm(zip(gsm8k_test['question'][:num_samples], gsm8k_test['answer'][:num_samples]),\n"
        "                         total=num_samples):": (
            "sample_end = args.sample_offset + num_samples\n"
            "        for q, a in tqdm(zip(gsm8k_test['question'][args.sample_offset:sample_end], "
            "gsm8k_test['answer'][args.sample_offset:sample_end]),\n"
            "                         total=num_samples):"
        ),
        'if args.prompt == "original":\n    prompt = open(\'lib_prompt/prompt_original.txt\').read()\nelse:\n    prompt = open(\'lib_prompt/prompt_hardest.txt\').read()': (
            'if args.prompt == "original":\n    prompt = (PROMPT_DIR / \'prompt_original.txt\').read_text()\nelse:\n    prompt = (PROMPT_DIR / \'prompt_hardest.txt\').read_text()\n\n'
            "if args.prompt_suffix:\n"
            "    print('prompt_suffix:', args.prompt_suffix.encode('unicode_escape').decode('ascii'))"
        ),
        '    if args.parallel:\n        sd += "_parallel"\n': (
            '    if args.parallel:\n        sd += "_parallel"\n\n'
            '    if args.prompt_tag:\n        sd += f"_{args.prompt_tag}"\n'
        ),
        "            input_text = tokenizer2.apply_chat_template(\n"
        "                messages,\n"
        "                tokenize=False,\n"
        "                add_generation_prompt=True\n"
        "            )\n": (
            "            input_text = tokenizer2.apply_chat_template(\n"
            "                messages,\n"
            "                tokenize=False,\n"
            "                add_generation_prompt=True\n"
            "            )\n"
            "\n"
            "            if args.prompt_suffix:\n"
            "                input_text += args.prompt_suffix\n"
        ),
        'total_counts["time"].append(start-end)': 'total_counts["time"].append(end-start)',
        # Flush after every sample so NFS sees partial results and they survive SIGTERM/preemption
        "            fd.write('Q: %s\\nA_model:\\n%s\\nA:\\n%s\\n\\n' % (q, ans_, a))\n": (
            "            fd.write('Q: %s\\nA_model:\\n%s\\nA:\\n%s\\n\\n' % (q, ans_, a))\n"
            "            fd.flush()  # flush after every sample so NFS sees progress and partial results survive SIGTERM\n"
        ),
        '                  "step_back_probs":[], "p_i":[], "q_i":[], "hist_lengths": [], "time":[], "ids":[]}': (
            '                  "step_back_probs":[], "p_i":[], "q_i":[], "hist_lengths": [], "time":[], "ids":[],\n'
            '                  "aggregate_acceptance_per_sample": [], "block_efficiency_per_sample": [],\n'
            '                  "accepted_draft_tokens_per_sample": [], "drafted_tokens_per_sample": [],\n'
            '                  "target_steps_per_sample": [], "first_token_acceptance_per_sample": [],\n'
            '                  "acceptance_trace_per_sample": [], "fallback_triggered_per_sample": [],\n'
            '                  "fallback_trigger_block_per_sample": [], "fallback_generated_tokens_per_sample": [],\n'
            '                  "generated_tokens_per_sample": []}'
        ),
        '                                                  return_probs=args.backward or args.blockwise,': '                                                  return_probs=True,',
        '                total_counts["sample_length"].append(len(outputs[0] - len(input_ids[0])))': (
            '                generated_length = int(outputs.shape[-1] - input_ids.shape[-1])\n'
            '                total_counts["sample_length"].append(generated_length)\n'
            '                total_counts["generated_tokens_per_sample"].append(generated_length)\n'
            '                total_counts["target_steps_per_sample"].append(float(generated_length))'
        ),
        '                total_counts["ids"].append(counts["ids"])\n': (
            '                total_counts["ids"].append(counts["ids"])\n'
            '\n'
            '                drafted_tokens = float(sum(counts["draft_eval"]))\n'
            '                accepted_draft_tokens = float(sum(counts.get("accepted_draft_tokens", [\n'
            '                    min(int(d), max(int(s) - int(t), 0))\n'
            '                    for d, s, t in zip(counts["draft_eval"], counts["sample_length"], counts["target_eval"])\n'
            '                ])))\n'
            '                generated_tokens = float(sum(counts["sample_length"]))\n'
            '                target_steps = float(sum(counts["total_step"]))\n'
            '                aggregate_acceptance = accepted_draft_tokens / drafted_tokens if drafted_tokens > 0 else None\n'
            '                block_efficiency = generated_tokens / target_steps if target_steps > 0 else None\n'
            '\n'
            '                first_token_acceptance = None\n'
            '                if counts["p_i"] and counts["q_i"] and len(counts["p_i"]) > 0 and len(counts["q_i"]) > 0:\n'
            '                    valid_prob_pairs = []\n'
            '                    for p_val, q_val in zip(counts["p_i"], counts["q_i"]):\n'
            '                        if p_val is None or q_val is None:\n'
            '                            continue\n'
            '                        try:\n'
            '                            p_float = float(p_val)\n'
            '                            q_float = float(q_val)\n'
            '                        except (TypeError, ValueError):\n'
            '                            continue\n'
            '                        valid_prob_pairs.append((p_float, q_float))\n'
            '                    if valid_prob_pairs:\n'
            '                        p0, q0 = valid_prob_pairs[0]\n'
            '                        first_token_acceptance = min(1.0, p0 / max(q0, 1e-12))\n'
            '\n'
            '                total_counts["accepted_draft_tokens_per_sample"].append(accepted_draft_tokens)\n'
            '                total_counts["drafted_tokens_per_sample"].append(drafted_tokens)\n'
            '                total_counts["target_steps_per_sample"].append(target_steps)\n'
            '                total_counts["aggregate_acceptance_per_sample"].append(aggregate_acceptance)\n'
            '                total_counts["block_efficiency_per_sample"].append(block_efficiency)\n'
            '                total_counts["first_token_acceptance_per_sample"].append(first_token_acceptance)\n'
            '                total_counts["acceptance_trace_per_sample"].append(counts.get("acceptance_trace", []))\n'
            '                total_counts["fallback_triggered_per_sample"].append(bool(counts.get("fallback_triggered", False)))\n'
            '                total_counts["fallback_trigger_block_per_sample"].append(counts.get("fallback_trigger_block"))\n'
            '                total_counts["fallback_generated_tokens_per_sample"].append(int(counts.get("fallback_generated_tokens", 0)))\n'
        ),
        '        with open(f"{sd}_total_counts.json", "w") as f:\n            json.dump(total_counts, f)\n': (
            '        with open(f"{sd}_total_counts.json", "w") as f:\n'
            '            valid_acceptance = [x for x in total_counts["aggregate_acceptance_per_sample"] if x is not None]\n'
            '            valid_block_eff = [x for x in total_counts["block_efficiency_per_sample"] if x is not None]\n'
            '            valid_first_alpha = [x for x in total_counts["first_token_acceptance_per_sample"] if x is not None]\n'
            '\n'
            '            total_counts["summary_metrics"] = {\n'
            '                "aggregate_acceptance_mean": float(np.mean(valid_acceptance)) if valid_acceptance else None,\n'
            '                "block_efficiency_mean": float(np.mean(valid_block_eff)) if valid_block_eff else None,\n'
            '                "first_token_acceptance_mean": float(np.mean(valid_first_alpha)) if valid_first_alpha else None,\n'
            '            }\n'
            '            json.dump(total_counts, f)\n'
        ),
    }

    for old, new in replacements.items():
        text = replace_once_or_skip(text, old, new, path)

    text = text.replace("max_new_tokens=512", "max_new_tokens=args.max_new_tokens")
    text = text.replace(
        "max_new_tokens=args.max_new_tokens,",
        "max_new_tokens=args.max_new_tokens,\n                                                  min_new_tokens=args.min_new_tokens,",
    )
    path.write_text(text)


def patch_generation_utils(path: Path) -> None:
    text = path.read_text()

    fallback_init = (
        '        fallback_threshold_raw = os.environ.get("ADSD_ACCEPTANCE_FALLBACK_THRESHOLD", "").strip()\n'
        '        fallback_threshold = float(fallback_threshold_raw) if fallback_threshold_raw else None\n'
        '        fallback_window = int(os.environ.get("ADSD_ACCEPTANCE_FALLBACK_WINDOW", "8"))\n'
        '        if fallback_threshold is not None and not 0.0 <= fallback_threshold <= 1.0:\n'
        '            raise ValueError("ADSD_ACCEPTANCE_FALLBACK_THRESHOLD must be in [0, 1]")\n'
        '        if fallback_window <= 0:\n'
        '            raise ValueError("ADSD_ACCEPTANCE_FALLBACK_WINDOW must be positive")\n'
        '        fallback_active = False\n'
        '        fallback_trigger_block = None\n'
        '        fallback_generated_tokens = 0\n'
        '        acceptance_trace = []\n'
        '        acceptance_window = []\n'
        '\n'
        '        counts = {"draft_eval": [], "target_eval": [], "total_step": [], "sample_length": [],\n'
        '                  "step_back_probs": [], "p_i": [], "q_i": [], "hist_lengths": [], "ids": [],\n'
        '                  "accepted_draft_tokens": [], "acceptance_trace": acceptance_trace,\n'
        '                  "fallback_threshold": fallback_threshold, "fallback_window": fallback_window,\n'
        '                  "fallback_triggered": False, "fallback_trigger_block": None,\n'
        '                  "fallback_generated_tokens": 0}'
    )
    replacements = {
        "                if backward:\n": (
            "                step_back_probs, p_next, q_next, ids = None, None, None, None\n"
            "                if backward:\n"
        ),
        "        if not return_probs:\n            return valid_tokens, n_matches, ind\n        else:\n            return valid_tokens, n_matches, None, None, None, None, ind\n": (
            "        if not return_probs:\n"
            "            return valid_tokens, n_matches, ind\n"
            "        else:\n"
            "            return valid_tokens, n_matches, None, p_i.cpu().numpy().tolist(), q_i.cpu().numpy().tolist(), "
            "new_candidate_input_ids.cpu().numpy().tolist(), ind\n"
        ),
        '        counts = {"draft_eval": [], "target_eval": [], "total_step": [], "sample_length": [],\n'
        '                  "step_back_probs": [], "p_i": [], "q_i": [], "hist_lengths": [], "ids": []}': (
            fallback_init
        ),
        "            num_residual_tokens = num_assistant_tokens\n": (
            "            if fallback_active:\n"
            "                candidate_generator.num_assistant_tokens = 0\n"
            "                num_assistant_tokens = 0\n"
            "\n"
            "            num_residual_tokens = num_assistant_tokens\n"
        ),
        "                candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)\n"
        "                # update the num_assistant_tokens if needed\n"
        "                num_assistant_tokens = candidate_generator.num_assistant_tokens\n": (
            "                candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)\n"
            "                # update the num_assistant_tokens if needed\n"
            "                num_assistant_tokens = candidate_generator.num_assistant_tokens\n"
            "                if fallback_active:\n"
            "                    candidate_generator.num_assistant_tokens = 0\n"
            "                    num_assistant_tokens = 0\n"
        ),
        '            counts["hist_lengths"].append(hist_lengths)\n': (
            '            counts["hist_lengths"].append(hist_lengths)\n'
            '            accepted_draft_tokens = min(int(draft_eval), max(int(sample_length) - int(target_eval), 0))\n'
            '            counts["accepted_draft_tokens"].append(accepted_draft_tokens)\n'
            '            if fallback_active:\n'
            '                fallback_generated_tokens += int(sample_length)\n'
            '            elif draft_eval > 0:\n'
            '                block_acceptance = float(accepted_draft_tokens) / float(draft_eval)\n'
            '                acceptance_trace.append(block_acceptance)\n'
            '                acceptance_window.append(block_acceptance)\n'
            '                if len(acceptance_window) > fallback_window:\n'
            '                    acceptance_window.pop(0)\n'
            '                if (\n'
            '                    fallback_threshold is not None\n'
            '                    and len(acceptance_window) == fallback_window\n'
            '                    and sum(acceptance_window) / len(acceptance_window) < fallback_threshold\n'
            '                ):\n'
            '                    fallback_active = True\n'
            '                    fallback_trigger_block = len(counts["total_step"])\n'
            '                    counts["fallback_triggered"] = True\n'
            '                    counts["fallback_trigger_block"] = fallback_trigger_block\n'
            '                    candidate_generator.num_assistant_tokens = 0\n'
            '                    num_assistant_tokens = 0\n'
            '            counts["fallback_generated_tokens"] = fallback_generated_tokens\n'
        ),
    }

    for old, new in replacements.items():
        text = replace_once_or_skip(text, old, new, path)
    path.write_text(text)


def ensure_outputs_dir(workspace: Path) -> None:
    outputs_dir = workspace / "chain-of-thought-hub" / "gsm8k" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a patched local HSD workspace")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    args = parser.parse_args()

    if not SOURCE_REPO.exists():
        raise SystemExit(f"Missing source repo: {SOURCE_REPO}")

    copy_workspace(SOURCE_REPO, args.workspace)
    patch_eval_script(args.workspace / "chain-of-thought-hub" / "gsm8k" / "eval_speculative_decoding_llm.py")
    patch_generation_utils(args.workspace / "transformers" / "generation" / "utils.py")
    ensure_outputs_dir(args.workspace)
    print(args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
