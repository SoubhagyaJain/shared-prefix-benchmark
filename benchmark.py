"""Controlled, sequential Ollama shared-prefix benchmark. No third-party packages."""

import argparse
import csv
import json
import platform
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from breakeven import summarize_break_even
from chart import save_break_even_chart, save_sweep_chart
from config import Config
from metrics import summarize, summarize_sweep
from prompts import make_prefix, prompt


CSV_FIELDS = [
    "run_id", "prefix_lines", "scenario", "trial", "request_index", "prompt_tokens", "cached_prompt_tokens",
    "uncached_prompt_tokens",
    "output_tokens", "ttft_s", "prefill_s", "generation_s", "api_total_s",
    "wall_latency_s", "prompt_tokens_per_s", "generation_tokens_per_s",
    "gpu_memory_mib", "load_s", "done_reason",
]


def ns_seconds(value):
    return value / 1e9 if isinstance(value, (int, float)) else None


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator and denominator > 0 else None


def api_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_version(config: Config) -> str | None:
    try:
        with urllib.request.urlopen(config.url + "/api/version", timeout=5) as response:
            return json.load(response).get("version")
    except (urllib.error.URLError, TimeoutError):
        return None


def gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def payload(config: Config, content: str) -> dict:
    return {
        "model": config.model,
        "prompt": content,
        "raw": True,
        "stream": True,
        "keep_alive": config.keep_alive,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_output_tokens,
            "num_ctx": config.context_window,
        },
    }


def stream_generate(config: Config, content: str) -> dict:
    request = urllib.request.Request(
        config.url + "/api/generate",
        data=json.dumps(payload(config, content)).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    first_token_at = None
    final = None
    with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
        for line in response:
            event = json.loads(line)
            if event.get("response") and first_token_at is None:
                first_token_at = time.perf_counter()
            if event.get("done"):
                final = event
    end = time.perf_counter()
    if final is None:
        raise RuntimeError("Ollama stream ended without final timing record")
    return {
        "prompt_tokens": final.get("prompt_eval_count"),
        "cached_prompt_tokens": final.get("prompt_eval_cached_count"),
        "output_tokens": final.get("eval_count"),
        "ttft_s": first_token_at - start if first_token_at is not None else None,
        "prefill_s": ns_seconds(final.get("prompt_eval_duration")),
        "generation_s": ns_seconds(final.get("eval_duration")),
        "api_total_s": ns_seconds(final.get("total_duration")),
        "wall_latency_s": end - start,
        "load_s": ns_seconds(final.get("load_duration")),
        "done_reason": final.get("done_reason"),
    }


def measured_request(config: Config, content: str) -> dict:
    row = stream_generate(config, content)
    # On Ollama 0.34.2, prompt_eval_count is full prompt length and
    # prompt_eval_cached_count is the number restored from cache.
    if row["prompt_tokens"] is not None and row["cached_prompt_tokens"] is not None:
        row["uncached_prompt_tokens"] = max(0, row["prompt_tokens"] - row["cached_prompt_tokens"])
    else:
        row["uncached_prompt_tokens"] = None
    row["prompt_tokens_per_s"] = ratio(row["uncached_prompt_tokens"], row["prefill_s"])
    row["generation_tokens_per_s"] = ratio(row["output_tokens"], row["generation_s"])
    row["gpu_memory_mib"] = gpu_memory_mib()  # end-of-request snapshot; not peak
    return row


def verify_context(prompt_tokens: int | None, config: Config, description: str, reserve: int = 0) -> None:
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise RuntimeError(f"{description}: Ollama did not report a valid prompt token count")
    required = prompt_tokens + config.max_output_tokens + reserve
    if required > config.context_window:
        raise RuntimeError(
            f"{description}: {prompt_tokens} prompt + {config.max_output_tokens} output"
            f" + {reserve} reserve = {required} tokens exceeds num_ctx={config.context_window}"
        )


def fmt(value, unit="") -> str:
    return "n/a" if value is None else f"{value:.3f}{unit}"


def print_summary(stats: dict) -> None:
    print("\nSHARED-PREFIX BENCHMARK")
    for key, label in (("different_prefix", "Different prefixes"), ("shared_prefix", "Shared prefix")):
        group = stats[key]
        print(f"\n{label}")
        for field, name, unit in (
            ("wall_latency_s", "Median wall latency", " s"),
            ("ttft_s", "Median TTFT", " s"),
            ("prefill_s", "Median prompt eval", " s"),
            ("cached_prompt_tokens", "Median cached prompt tokens", ""),
            ("prompt_tokens_per_s", "Median uncached prompt rate", " tok/s"),
        ):
            print(f"  {name}: {fmt(group[field]['median'], unit)}")
        print(f"  Request throughput: {fmt(group['request_throughput_per_s'], ' req/s')}")
    print("\nDifference (positive favors shared)")
    for key, label in (("prefill_median", "Prompt eval"), ("latency_median", "Wall latency"),
                       ("request_throughput", "Request throughput")):
        print(f"  {label}: {fmt(stats['improvement_percent'][key], '%')}")
    if stats["shared_prefix"]["cached_prompt_tokens"]["count"] == 0:
        print("Cached-token count unavailable; timings alone cannot establish cache reuse.")


def parse_prefix_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prefix lengths must be comma-separated positive integers") from exc
    if not lengths or any(length <= 0 for length in lengths) or len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("prefix lengths must be unique positive integers")
    return lengths


def parse_args(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=defaults.url)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--trials", type=int, default=defaults.trials)
    parser.add_argument("--requests", type=int, default=defaults.requests_per_trial)
    parser.add_argument("--prefix-lines", type=int, default=defaults.prefix_lines)
    parser.add_argument("--sweep", action="store_true", help="run all configured prefix lengths")
    parser.add_argument("--break-even", action="store_true", help="measure cold-prefix reuse break-even")
    parser.add_argument("--max-requests", type=int,
                        help="maximum requests per scenario per trial; implies --break-even (default: 8)")
    parser.add_argument("--prefix-lengths", type=parse_prefix_lengths,
                        help="comma-separated sweep lengths (default: 20,40,80,105,160,220); implies --sweep")
    parser.add_argument("--max-output-tokens", type=int, default=defaults.max_output_tokens)
    parser.add_argument("--num-ctx", type=int, default=defaults.context_window)
    parser.add_argument("--timeout", type=int, default=defaults.timeout_s)
    parser.add_argument("--csv", help="request-level CSV output path")
    parser.add_argument("--summary", help="JSON summary output path")
    parser.add_argument("--cumulative-csv", help="break-even cumulative CSV output path")
    parser.add_argument("--chart", help="SVG chart output path")
    args = parser.parse_args(argv)
    if min(args.trials, args.requests, args.prefix_lines, args.max_output_tokens, args.num_ctx, args.timeout) <= 0:
        parser.error("counts and context size must be positive")
    sweep = args.sweep or args.prefix_lengths is not None
    break_even = args.break_even or args.max_requests is not None
    if sweep and break_even:
        parser.error("--sweep/--prefix-lengths and --break-even/--max-requests cannot be combined")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    if break_even:
        default_csv = "results/breakeven_raw_results.csv"
        default_summary = "results/breakeven_summary.json"
        default_chart = "results/breakeven_chart.svg"
    elif sweep:
        default_csv = "results/sweep_raw_results.csv"
        default_summary = "results/sweep_summary.json"
        default_chart = defaults.chart_svg
    else:
        default_csv = defaults.raw_csv
        default_summary = defaults.summary_json
        default_chart = defaults.chart_svg
    return Config(url=args.url.rstrip("/"), model=args.model, trials=args.trials,
                  requests_per_trial=args.requests, prefix_lines=args.prefix_lines,
                  max_output_tokens=args.max_output_tokens, context_window=args.num_ctx,
                  timeout_s=args.timeout,
                  raw_csv=args.csv or default_csv,
                  summary_json=args.summary or default_summary,
                  sweep=sweep, prefix_lengths=args.prefix_lengths or defaults.prefix_lengths,
                  chart_svg=args.chart or default_chart,
                  break_even=break_even,
                  max_requests_per_trial=args.max_requests or defaults.max_requests_per_trial,
                  cumulative_csv=args.cumulative_csv or defaults.cumulative_csv)


def run_break_even(config: Config) -> int:
    run_id = secrets.token_hex(12)
    version = get_version(config)
    if version is None:
        print(f"Ollama is unavailable at {config.url}; start Ollama first.", file=sys.stderr)
        return 1
    print(f"Ollama {version}; model {config.model}; sequential cold-prefix trials")
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    csv_path = Path(config.raw_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[*CSV_FIELDS, "prefix_state"])
        writer.writeheader()
        csv_file.flush()
        for trial in range(1, config.trials + 1):
            # Load and warm the model with unrelated text before either measured sequence.
            for index in range(2):
                stream_generate(config, f"Trial warm-up {run_id}-{trial}-{index}: Say hello briefly.")
            warmup = stream_generate(
                config, prompt(make_prefix(f"warmup-{run_id}-{trial:03d}", config.prefix_lines), 999)
            )
            verify_context(warmup["prompt_tokens"], config, f"trial {trial} warm-up", reserve=256)

            shared = make_prefix(f"shared-{run_id}-{trial:03d}-000", config.prefix_lines)
            order = ("different_prefix", "shared_prefix") if trial % 2 else ("shared_prefix", "different_prefix")
            for scenario in order:
                for index in range(1, config.max_requests_per_trial + 1):
                    if scenario == "shared_prefix":
                        prefix = shared
                        state = "cold" if index == 1 else "warm"
                    else:
                        prefix = make_prefix(f"unique-{run_id}-{trial:03d}-{index:03d}", config.prefix_lines)
                        state = "new"
                    result = measured_request(config, prompt(prefix, index))
                    verify_context(result["prompt_tokens"], config,
                                   f"trial {trial} {scenario} request {index}", reserve=256)
                    row = {"run_id": run_id, "prefix_lines": config.prefix_lines,
                           "scenario": scenario, "trial": trial, "request_index": index,
                           "prefix_state": state, **result}
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    print(f"trial={trial} {scenario} request={index} state={state} "
                          f"tokens={result['prompt_tokens']} cached={result['cached_prompt_tokens']} "
                          f"prefill={fmt(result['prefill_s'], 's')} wall={fmt(result['wall_latency_s'], 's')}",
                          flush=True)

    cumulative = summarize_break_even(rows, config.trials, config.max_requests_per_trial)
    cumulative_path = Path(config.cumulative_csv)
    cumulative_path.parent.mkdir(parents=True, exist_ok=True)
    points = [*cumulative["per_trial_curves"], *cumulative["by_request_count"].values()]
    fields = list(dict.fromkeys(key for point in points for key in point))
    with cumulative_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)

    summary = {
        "started_utc": started,
        "run_id": run_id,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_version": version,
        "model": config.model,
        "system": platform.platform(),
        "settings": {"trials": config.trials, "max_requests_per_trial": config.max_requests_per_trial,
                     "prefix_lines": config.prefix_lines, "temperature": config.temperature,
                     "max_output_tokens": config.max_output_tokens, "num_ctx": config.context_window,
                     "concurrency": 1, "raw_prompt": True},
        "notes": [
            "Cold means an unseen prefix with an already loaded and warmed model; each trial has unrelated unmeasured model warm-ups.",
            "The first shared-prefix request is measured without priming. Later shared requests use distinct questions on that prefix.",
            "Distinct-prefix requests use new similarly sized reference sets at each index.",
            "Scenario order alternates by trial. All measured requests are sequential and streamed.",
            "Break-even is the first N with strictly lower mean cumulative shared time than distinct time; priming and warm-up are excluded from cumulative totals.",
            "Prompt counts are reported by Ollama and checked with room for max output tokens plus a 256-token reserve.",
        ],
        **cumulative,
    }
    summary_path = Path(config.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_break_even_chart(cumulative["by_request_count"], config.chart_svg)
    for label, key in (("Wall-latency", "wall_first_n"), ("Prompt-evaluation", "prompt_eval_first_n")):
        point = cumulative["break_even"][key]
        print(f"{label} break-even: N={point}" if point is not None else
              f"{label} break-even: not reached through N={config.max_requests_per_trial}")
    print(f"Saved {csv_path}, {cumulative_path}, {summary_path}, and {config.chart_svg}")
    return 0


def main() -> int:
    config = parse_args()
    if config.break_even:
        return run_break_even(config)
    lengths = config.prefix_lengths if config.sweep else (config.prefix_lines,)
    run_id = secrets.token_hex(12)
    version = get_version(config)
    if version is None:
        print(f"Ollama is unavailable at {config.url}; start Ollama first.", file=sys.stderr)
        return 1
    print(f"Ollama {version}; model {config.model}; sequential requests")
    print("Warming model and runner...")
    for index in range(2):
        stream_generate(config, f"Warm-up {index}: Say hello in one short sentence.")
    longest = max(lengths)
    warmup = stream_generate(config, prompt(make_prefix(f"long-warmup-{run_id}", longest), 999))
    # Ollama has no standard tokenize endpoint. A distinct longest-length probe
    # provides a preflight budget; every subsequent API count is checked too.
    verify_context(warmup["prompt_tokens"], config, "long warm-up", reserve=256)

    rows = []
    csv_path = Path(config.raw_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        block_number = 0
        for length in lengths:
            shared = make_prefix(f"shared-{run_id}-{length}", length)
            for trial in range(1, config.trials + 1):
                # Alternate order across all length/trial pairs.
                order = ("different_prefix", "shared_prefix") if block_number % 2 == 0 else ("shared_prefix", "different_prefix")
                block_number += 1
                for scenario in order:
                    if scenario == "shared_prefix":
                        # Prime outside the measured rows.
                        prime = stream_generate(config, prompt(shared, 900 + trial))
                        verify_context(prime["prompt_tokens"], config, f"{length}-line shared prime", reserve=256)
                    for index in range(1, config.requests_per_trial + 1):
                        identity = f"unique-{run_id}-{length}-{trial:03d}-{index:03d}"
                        prefix = shared if scenario == "shared_prefix" else make_prefix(identity, length)
                        result = measured_request(config, prompt(prefix, index))
                        verify_context(result["prompt_tokens"], config, f"{length}-line {scenario} request", reserve=256)
                        row = {"run_id": run_id, "prefix_lines": length, "scenario": scenario,
                               "trial": trial, "request_index": index, **result}
                        rows.append(row)
                        writer.writerow(row)
                        csv_file.flush()
                        print(f"lines={length} {scenario} trial={trial} request={index} "
                              f"tokens={result['prompt_tokens']} cached={result['cached_prompt_tokens']} "
                              f"prefill={fmt(result['prefill_s'], 's')} wall={fmt(result['wall_latency_s'], 's')}", flush=True)

    stats = summarize_sweep(rows, lengths) if config.sweep else summarize(rows)
    summary_path = Path(config.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "started_utc": started,
        "run_id": run_id,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_version": version,
        "model": config.model,
        "system": platform.platform(),
        "settings": {"trials": config.trials, "requests_per_trial": config.requests_per_trial,
                     **({"prefix_lengths": list(lengths)} if config.sweep else {"prefix_lines": config.prefix_lines}),
                     "temperature": config.temperature,
                     "max_output_tokens": config.max_output_tokens, "num_ctx": config.context_window,
                     "concurrency": 1, "raw_prompt": True},
        "notes": ["The reported workload uses a warm prefix. Shared prefix is explicitly primed before each block; priming requests are excluded from measurements.",
                  "Prompt token counts are reported by Ollama; the distinct longest-length warm-up and each priming and measured request are checked against num_ctx with room for max output tokens and a 256-token reserve.",
                  "GPU memory is an end-of-request whole-device snapshot, not peak or model-only usage.",
                  "TTFT is client-observed time to first nonempty streamed response chunk.",
                  "Prompt tokens/s divides uncached prompt tokens by prompt evaluation duration; unavailable without cached-token count."],
        "statistics": stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.sweep:
        save_sweep_chart(stats, lengths, config.chart_svg)
        for length in lengths:
            gain = stats[str(length)]["improvement_percent"]
            print(f"{length} lines: median TTFT {fmt(gain['ttft_s']['median'], '%')} improvement; "
                  f"median prompt eval {fmt(gain['prefill_s']['median'], '%')} improvement")
        print(f"\nSaved {csv_path}, {summary_path}, and {config.chart_svg}")
    else:
        print_summary(stats)
        print(f"\nSaved {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"Ollama request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
