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

from config import Config
from metrics import summarize
from prompts import make_prefix, prompt


CSV_FIELDS = [
    "scenario", "trial", "request_index", "prompt_tokens", "cached_prompt_tokens",
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


def parse_args():
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=defaults.url)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--trials", type=int, default=defaults.trials)
    parser.add_argument("--requests", type=int, default=defaults.requests_per_trial)
    parser.add_argument("--prefix-lines", type=int, default=defaults.prefix_lines)
    parser.add_argument("--max-output-tokens", type=int, default=defaults.max_output_tokens)
    parser.add_argument("--num-ctx", type=int, default=defaults.context_window)
    parser.add_argument("--timeout", type=int, default=defaults.timeout_s)
    parser.add_argument("--csv", default=defaults.raw_csv)
    parser.add_argument("--summary", default=defaults.summary_json)
    args = parser.parse_args()
    if min(args.trials, args.requests, args.prefix_lines, args.max_output_tokens, args.num_ctx) <= 0:
        parser.error("counts and context size must be positive")
    return Config(url=args.url.rstrip("/"), model=args.model, trials=args.trials,
                  requests_per_trial=args.requests, prefix_lines=args.prefix_lines,
                  max_output_tokens=args.max_output_tokens, context_window=args.num_ctx,
                  timeout_s=args.timeout, raw_csv=args.csv, summary_json=args.summary)


def main() -> int:
    config = parse_args()
    run_id = secrets.token_hex(6)
    version = get_version(config)
    if version is None:
        print(f"Ollama is unavailable at {config.url}; start Ollama first.", file=sys.stderr)
        return 1
    print(f"Ollama {version}; model {config.model}; sequential requests")
    print("Warming model and runner...")
    for index in range(2):
        stream_generate(config, f"Warm-up {index}: Say hello in one short sentence.")
    stream_generate(config, prompt(make_prefix(f"long-warmup-{run_id}", config.prefix_lines), 999))

    shared = make_prefix(f"shared-{run_id}", config.prefix_lines)
    rows = []
    csv_path = Path(config.raw_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        for trial in range(1, config.trials + 1):
            # Alternate block order to limit monotonic clock/thermal effects.
            order = ("different_prefix", "shared_prefix") if trial % 2 else ("shared_prefix", "different_prefix")
            for scenario in order:
                if scenario == "shared_prefix":
                    # Explicitly prime the common prefix outside measured requests.
                    stream_generate(config, prompt(shared, 900 + trial))
                for index in range(1, config.requests_per_trial + 1):
                    identity = f"unique-{run_id}-{trial:03d}-{index:03d}"
                    prefix = shared if scenario == "shared_prefix" else make_prefix(identity, config.prefix_lines)
                    content = prompt(prefix, index)
                    result = measured_request(config, content)
                    row = {"scenario": scenario, "trial": trial, "request_index": index, **result}
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    print(f"{scenario} trial={trial} request={index} "
                          f"tokens={result['prompt_tokens']} cached={result['cached_prompt_tokens']} "
                          f"prefill={fmt(result['prefill_s'], 's')} wall={fmt(result['wall_latency_s'], 's')}", flush=True)

    stats = summarize(rows)
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
                     "prefix_lines": config.prefix_lines, "temperature": config.temperature,
                     "max_output_tokens": config.max_output_tokens, "num_ctx": config.context_window,
                     "concurrency": 1, "raw_prompt": True},
        "notes": ["Shared prefix is explicitly primed before each block.",
                  "GPU memory is an end-of-request whole-device snapshot, not peak or model-only usage.",
                  "TTFT is client-observed time to first nonempty streamed response chunk.",
                  "Prompt tokens/s divides uncached prompt tokens by prompt evaluation duration; unavailable without cached-token count."],
        "statistics": stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(stats)
    print(f"\nSaved {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"Ollama request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
